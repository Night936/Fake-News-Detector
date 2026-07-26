"""
Extension of the original models/arg.py ARGModel.

Structural changes from the original ARG model:
  1. A small MLP projects the 12-dim VADER+lexical feature vector
     (extra_features) into a 64-dim embedding.
  2. A small MLP projects the 8-dim comment volume/sentiment/engagement
     vector (comment_features -- the "social branch": see
     Utils/commentFeatures.py and Preprocessing/commentAggregation.py) into
     a comment_feature_mlp_dim embedding.
Both get concatenated onto the BERT+rationale final_feature right before the
classification head. This keeps the entire rationale-fusion mechanism
(MaskAttention aggregator, co-attention, cross-attention, gating) exactly as
ARG's authors designed it -- we're only widening the final decision layer's
input, not touching how text and rationales talk to each other.

SOCIAL BRANCH FIX (see project write-up for full detail): the previous
version of this file had a second, broken attempt at using comments that
tokenized a "comment digest" via undefined `comment_ids` / `comment_masks`
variables, called `F.cosine_similarity` without importing
`torch.nn.functional as F`, declared `self.mlp` twice (silently discarding
the first, correctly-sized definition), and then overwrote `fused_feature`
right before the classifier call in a way that dropped the comment embedding
it had just computed -- so the branch could never construct or run. That
code has been removed. What remains below is the single, working numeric
comment-feature fusion path (mirrors the extra_features pattern, which
already worked), wired all the way from config -> dataloader -> model.
"""

import os
import time

import torch
import torch.nn as nn
import tqdm
from transformers import BertModel

from Models.layers import MLP, MaskAttention, ParallelCoAttentionNetwork, SelfAttentionFeatureExtract
from Utils.fakedditDataloader import data2gpu, get_dataloader
from Utils.argUtils import Averager, Recorder, metrics


class ARGFakedditModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert_content = BertModel.from_pretrained(config["bert_path"]).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(config["bert_path"]).requires_grad_(False)

        for bert in (self.bert_content, self.bert_FTR):
            for name, param in bert.named_parameters():
                param.requires_grad = name.startswith("encoder.layer.11")

        emb_dim = config["emb_dim"]
        mlp_dims = config["model"]["mlp"]["dims"]
        mlp_dropout = config["model"]["mlp"]["dropout"]

        self.aggregator = MaskAttention(emb_dim)

        # extra (sentiment + lexical) fusion branch
        extra_dim = config["extra_feature_dim"]
        extra_mlp_dim = config["extra_feature_mlp_dim"]
        self.extra_feature_mlp = nn.Sequential(
            nn.Linear(extra_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, extra_mlp_dim),
            nn.ReLU(),
        )

        # social / comment-engagement fusion branch ("social branch")
        comment_dim = config["comment_feature_dim"]          # 8, from Utils/commentFeatures.py
        comment_mlp_dim = config["comment_feature_mlp_dim"]  # e.g. 32
        self.comment_feature_mlp = nn.Sequential(
            nn.Linear(comment_dim, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, comment_mlp_dim),
            nn.ReLU(),
        )

        # classification head takes [final_feature ; extra_embedding ; comment_embedding]
        # (single definition -- the previous duplicate `self.mlp = ...` that
        # silently overrode this with a mismatched size has been removed)
        self.mlp = MLP(emb_dim + extra_mlp_dim + comment_mlp_dim, mlp_dims, mlp_dropout)

        self.hard_ftr_2_attention = MaskAttention(emb_dim)
        self.hard_mlp_ftr_2 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dims[-1]), nn.ReLU(), nn.Linear(mlp_dims[-1], 1), nn.Sigmoid()
        )
        self.score_mapper_ftr_2 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dims[-1]), nn.BatchNorm1d(mlp_dims[-1]), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(mlp_dims[-1], 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

        self.hard_ftr_3_attention = MaskAttention(emb_dim)
        self.hard_mlp_ftr_3 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dims[-1]), nn.ReLU(), nn.Linear(mlp_dims[-1], 1), nn.Sigmoid()
        )
        self.score_mapper_ftr_3 = nn.Sequential(
            nn.Linear(emb_dim, mlp_dims[-1]), nn.BatchNorm1d(mlp_dims[-1]), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(mlp_dims[-1], 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

        self.simple_ftr_2_attention = MaskAttention(emb_dim)
        self.simple_mlp_ftr_2 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]), nn.ReLU(), nn.Linear(mlp_dims[-1], 3))
        self.simple_ftr_3_attention = MaskAttention(emb_dim)
        self.simple_mlp_ftr_3 = nn.Sequential(nn.Linear(emb_dim, mlp_dims[-1]), nn.ReLU(), nn.Linear(mlp_dims[-1], 3))

        self.content_attention = MaskAttention(emb_dim)

        self.co_attention_2 = ParallelCoAttentionNetwork(emb_dim, config["co_attention_dim"], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(emb_dim, config["co_attention_dim"], mask_in=True)

        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, emb_dim)
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, emb_dim)
        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, emb_dim)
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, emb_dim)

    def forward(self, **kwargs):
        content, content_masks = kwargs["content"], kwargs["content_masks"]
        FTR_2, FTR_2_masks = kwargs["FTR_2"], kwargs["FTR_2_masks"]
        FTR_3, FTR_3_masks = kwargs["FTR_3"], kwargs["FTR_3_masks"]
        extra_features = kwargs["extra_features"]
        comment_features = kwargs["comment_features"]

        content_feature = self.bert_content(content, attention_mask=content_masks)[0]
        FTR_2_feature = self.bert_FTR(FTR_2, attention_mask=FTR_2_masks)[0]
        FTR_3_feature = self.bert_FTR(FTR_3, attention_mask=FTR_3_masks)[0]

        mutual_content_FTR_2, _ = self.cross_attention_content_2(content_feature, FTR_2_feature, content_masks)
        expert_2 = torch.mean(mutual_content_FTR_2, dim=1)

        mutual_content_FTR_3, _ = self.cross_attention_content_3(content_feature, FTR_3_feature, content_masks)
        expert_3 = torch.mean(mutual_content_FTR_3, dim=1)

        mutual_FTR_content_2, _ = self.cross_attention_ftr_2(FTR_2_feature, content_feature, FTR_2_masks)
        mutual_FTR_content_2 = torch.mean(mutual_FTR_content_2, dim=1)

        mutual_FTR_content_3, _ = self.cross_attention_ftr_3(FTR_3_feature, content_feature, FTR_3_masks)
        mutual_FTR_content_3 = torch.mean(mutual_FTR_content_3, dim=1)

        hard_ftr_2_pred = self.hard_mlp_ftr_2(mutual_FTR_content_2).squeeze(1)
        hard_ftr_3_pred = self.hard_mlp_ftr_3(mutual_FTR_content_3).squeeze(1)

        simple_ftr_2_pred = self.simple_mlp_ftr_2(self.simple_ftr_2_attention(FTR_2_feature)[0]).squeeze(1)
        simple_ftr_3_pred = self.simple_mlp_ftr_3(self.simple_ftr_3_attention(FTR_3_feature)[0]).squeeze(1)

        attn_content, _ = self.content_attention(content_feature, mask=content_masks)

        reweight_score_ftr_2 = self.score_mapper_ftr_2(mutual_FTR_content_2)
        reweight_score_ftr_3 = self.score_mapper_ftr_3(mutual_FTR_content_3)
        reweight_expert_2 = reweight_score_ftr_2 * expert_2
        reweight_expert_3 = reweight_score_ftr_3 * expert_3

        all_feature = torch.cat(
            (attn_content.unsqueeze(1), reweight_expert_2.unsqueeze(1), reweight_expert_3.unsqueeze(1)), dim=1
        )
        final_feature, _ = self.aggregator(all_feature)

        extra_embedding = self.extra_feature_mlp(extra_features)
        comment_embedding = self.comment_feature_mlp(comment_features)
        fused_feature = torch.cat([final_feature, extra_embedding, comment_embedding], dim=1)

        label_pred = self.mlp(fused_feature)
        gate_value = torch.cat([reweight_score_ftr_2, reweight_score_ftr_3], dim=1)

        return {
            "classify_pred": torch.sigmoid(label_pred.squeeze(1)),
            "gate_value": gate_value,
            "final_feature": final_feature,
            "fused_feature": fused_feature,
            "hard_ftr_2_pred": hard_ftr_2_pred,
            "hard_ftr_3_pred": hard_ftr_3_pred,
            "simple_ftr_2_pred": simple_ftr_2_pred,
            "simple_ftr_3_pred": simple_ftr_3_pred,
        }


class Trainer:
    def __init__(self, config):
        self.config = config
        self.num_expert = 2
        self.save_path = os.path.join(config["param_dir"], "text_branch")
        os.makedirs(self.save_path, exist_ok=True)

    def train(self, logger=None):
        cfg = self.config
        self.model = ARGFakedditModel(cfg)
        if cfg["use_cuda"]:
            self.model = self.model.cuda()

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        recorder = Recorder(cfg["early_stop"])

        train_loader = get_dataloader(
            cfg["train_path"], cfg["max_len"], cfg["batchsize"], True,
            cfg["bert_path"], cfg["extra_feature_dim"], cfg["comment_feature_dim"],
        )
        val_loader = get_dataloader(
            cfg["val_path"], cfg["max_len"], cfg["batchsize"], False,
            cfg["bert_path"], cfg["extra_feature_dim"], cfg["comment_feature_dim"],
        )
        test_loader = get_dataloader(
            cfg["test_path"], cfg["max_len"], cfg["batchsize"], False,
            cfg["bert_path"], cfg["extra_feature_dim"], cfg["comment_feature_dim"],
        )

        for epoch in range(cfg["epoch"]):
            print(f"---------- epoch {epoch} ----------")
            self.model.train()
            avg_loss = Averager()

            for batch in tqdm.tqdm(train_loader):
                batch_data = data2gpu(batch, cfg["use_cuda"])
                label = batch_data["label"]

                res = self.model(**batch_data)
                loss_classify = loss_fn(res["classify_pred"], label.float())

                loss_hard = nn.BCELoss()(res["hard_ftr_2_pred"], batch_data["FTR_2_acc"].float()) + \
                    nn.BCELoss()(res["hard_ftr_3_pred"], batch_data["FTR_3_acc"].float())
                loss_simple = nn.CrossEntropyLoss()(res["simple_ftr_2_pred"], batch_data["FTR_2_pred"].long()) + \
                    nn.CrossEntropyLoss()(res["simple_ftr_3_pred"], batch_data["FTR_3_pred"].long())

                loss = loss_classify
                loss += cfg["model"]["rationale_usefulness_evaluator_weight"] * loss_hard / self.num_expert
                loss += cfg["model"]["llm_judgment_predictor_weight"] * loss_simple / self.num_expert

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss_classify.item())

            print("----- validating -----")
            val_results = self.evaluate(val_loader)
            print("val:", val_results)
            mark = recorder.add(val_results)
            if logger:
                logger.info(f"epoch {epoch} train_loss={avg_loss.item():.4f} val={val_results}")

            if mark == "save":
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "parameter_bert.pkl"))
            if mark == "esc":
                break

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, "parameter_bert.pkl")))
        test_results = self.evaluate(test_loader)
        print("test results:", test_results)
        return test_results, os.path.join(self.save_path, "parameter_bert.pkl")

    def evaluate(self, dataloader):
        self.model.eval()
        pred, label = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                batch_data = data2gpu(batch, self.config["use_cuda"])
                res = self.model(**batch_data)
                label.extend(batch_data["label"].detach().cpu().numpy().tolist())
                pred.extend(res["classify_pred"].detach().cpu().numpy().tolist())
        return metrics(label, pred)