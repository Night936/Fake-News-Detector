"""
Models/progressiveFusion.py

Progressive Multimodal Fusion Network (PFN) -- Jing et al. (2023)
"Multimodal fake news detection via progressive fusion networks", adapted to
this project's three existing signal sources: the ARG-style rationale text
branch, the dual spatial(Swin)/frequency(VGG19-on-DFT) image branch, and the
social/comment feature branch.

Difference from the earlier text-branch/image-branch approach (Models/argFakeddit.py,
Models/imageModel.py): instead of training each branch separately and concatenating
frozen final-layer embeddings, this fuses text, spatial, and frequency
representations at N_STAGES progressively deeper levels, carrying a running
fused state forward stage-to-stage. The paper does this with an Mlp-Mixer
block per stage; here a GRUCell-based update plays the same role (mix the
three modalities at this depth, update a running summary) while staying
light enough to train on this project's small multimodal subset.

Depth alignment across modalities:
  - text:    4 evenly spaced BERT hidden-state depths (via output_hidden_states=True)
  - spatial: Swin Transformer's 4 native stages (timm, features_only=True)
  - freq:    the last 4 of VGG19's 5 native stages, run on the 6-channel
             real+imag DFT tensor from Preprocessing/imagePreprocessing.py's
             getFrequencyComponents(), adapted to 3 input channels with a
             learned 1x1 conv (FreqAdapter) so VGG19's pretrained weights
             don't need to change.

extra_features (VADER/lexical) and comment_features (social/engagement) are
fused the same way they already are in Models/argFakeddit.py -- projected by
a small MLP each and concatenated onto the final fused state before the
classification head.

NOTE on auxiliary losses: ARG's rationale-usefulness / LLM-judgment auxiliary
heads (hard_mlp_ftr_2/3, simple_mlp_ftr_2/3 in Models/argFakeddit.py) are not
reproduced here yet -- this first working version optimizes classify loss
only. Re-adding them (computed once, e.g. off the deepest text stage) is a
natural next step once this trains cleanly end-to-end; it was left out to
keep this first pass mergeable and testable in isolation.
"""

import os

import timm
import torch
import torch.nn as nn
import tqdm
from transformers import BertModel

from Models.layers import MaskAttention, SelfAttentionFeatureExtract
from Utils.argUtils import Averager, Recorder, metrics
from Utils.imageDataloader import get_multimodal_dataloader

N_STAGES = 4


class FreqAdapter(nn.Module):
    """1x1 conv mapping the 6-channel real+imag DFT tensor down to 3
    channels, so a pretrained (3-channel) VGG19 backbone can be reused
    unmodified -- keeps VGG19's ImageNet weights intact; only this small
    adapter is trained from scratch."""

    def __init__(self):
        super().__init__()
        self.adapt = nn.Conv2d(6, 3, kernel_size=1)

    def forward(self, x):
        return self.adapt(x)


def _pool_stage(feat, expected_dim):
    """timm's Swin (features_only=True) can return channels-last
    (B, H, W, C) or channels-first (B, C, H, W) depending on version;
    this pools spatial dims correctly either way by checking which axis
    matches the projection layer's expected input size."""
    if feat.dim() == 4 and feat.shape[-1] == expected_dim:
        return feat.mean(dim=(1, 2))
    return feat.mean(dim=(2, 3))


class StageFusionBlock(nn.Module):
    """Fuses text/spatial/frequency embeddings at one depth stage and
    updates a running fused state carried from the previous stage."""

    def __init__(self, dim):
        super().__init__()
        self.text_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.spatial_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.freq_proj = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.mix = nn.Sequential(
            nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )
        self.update = nn.GRUCell(dim, dim)

    def forward(self, state, text_feat, spatial_feat, freq_feat):
        t = self.text_proj(text_feat)
        s = self.spatial_proj(spatial_feat)
        f = self.freq_proj(freq_feat)
        mixed = self.mix(torch.cat([t, s, f], dim=1))
        return self.update(mixed, state)


class ProgressiveFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["fusion_dim"]
        self.dim = dim

        # ---- text branch: BERT + ARG-style rationale cross-attention ----
        self.bert_content = BertModel.from_pretrained(
            config["bert_path"], output_hidden_states=True
        ).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(
            config["bert_path"], output_hidden_states=True
        ).requires_grad_(False)
        for bert in (self.bert_content, self.bert_FTR):
            for name, param in bert.named_parameters():
                param.requires_grad = name.startswith("encoder.layer.11")

        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, dim)
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, dim)
        self.content_attention = MaskAttention(dim)

        n_layers = self.bert_content.config.num_hidden_layers  # 12 for bert-base
        # 4 evenly spaced depths, skipping the raw embedding layer (index 0)
        self.text_stage_layers = [
            max(1, round(n_layers * (i + 1) / N_STAGES)) for i in range(N_STAGES)
        ]

        # ---- spatial branch: Swin Transformer, 4 native stages ----
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, features_only=True
        )
        swin_dims = self.swin.feature_info.channels()  # 4 stages, e.g. [96,192,384,768]
        self.swin_proj = nn.ModuleList([nn.Linear(d, dim) for d in swin_dims])

        # ---- frequency branch: VGG19 on the 6-channel real+imag DFT tensor ----
        self.freq_adapter = FreqAdapter()
        self.vgg = timm.create_model("vgg19", pretrained=True, features_only=True)
        vgg_dims = self.vgg.feature_info.channels()
        # VGG19 exposes 5 stages by default; use the deepest 4 to line up with Swin/text
        self.vgg_stage_idx = list(range(len(vgg_dims)))[-N_STAGES:]
        self.vgg_proj = nn.ModuleList([nn.Linear(vgg_dims[i], dim) for i in self.vgg_stage_idx])

        self.stages = nn.ModuleList([StageFusionBlock(dim) for _ in range(N_STAGES)])
        self.init_state = nn.Parameter(torch.zeros(1, dim))

        # extra (sentiment/lexical) + comment (social) fusion -- same
        # pattern as Models/argFakeddit.py
        extra_dim, extra_mlp_dim = config["extra_feature_dim"], config["extra_feature_mlp_dim"]
        comment_dim, comment_mlp_dim = config["comment_feature_dim"], config["comment_feature_mlp_dim"]
        self.extra_feature_mlp = nn.Sequential(
            nn.Linear(extra_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, extra_mlp_dim), nn.ReLU(),
        )
        self.comment_feature_mlp = nn.Sequential(
            nn.Linear(comment_dim, 24), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(24, comment_mlp_dim), nn.ReLU(),
        )

        mlp_dims = config["model"]["mlp"]["dims"]
        mlp_dropout = config["model"]["mlp"]["dropout"]
        head_in = dim + extra_mlp_dim + comment_mlp_dim
        layers = []
        in_d = head_in
        for d in mlp_dims:
            layers += [nn.Linear(in_d, d), nn.ReLU(), nn.Dropout(mlp_dropout)]
            in_d = d
        layers.append(nn.Linear(in_d, 1))
        self.classifier = nn.Sequential(*layers)

    def _text_stage_features(self, content, content_masks, ftr2, ftr2_masks, ftr3, ftr3_masks):
        out_content = self.bert_content(content, attention_mask=content_masks)
        out_ftr2 = self.bert_FTR(ftr2, attention_mask=ftr2_masks)
        out_ftr3 = self.bert_FTR(ftr3, attention_mask=ftr3_masks)

        stage_feats = []
        for layer_idx in self.text_stage_layers:
            content_h = out_content.hidden_states[layer_idx]
            ftr2_h = out_ftr2.hidden_states[layer_idx]
            ftr3_h = out_ftr3.hidden_states[layer_idx]

            mutual2, _ = self.cross_attention_content_2(content_h, ftr2_h, content_masks)
            mutual3, _ = self.cross_attention_content_3(content_h, ftr3_h, content_masks)
            expert2 = torch.mean(mutual2, dim=1)
            expert3 = torch.mean(mutual3, dim=1)
            attn_content, _ = self.content_attention(content_h, mask=content_masks)
            stage_feats.append((attn_content + expert2 + expert3) / 3.0)
        return stage_feats

    def _spatial_stage_features(self, img_spatial):
        feats = self.swin(img_spatial)
        return [proj(_pool_stage(f, proj.in_features)) for proj, f in zip(self.swin_proj, feats)]

    def _freq_stage_features(self, img_freq_complex):
        x = self.freq_adapter(img_freq_complex)
        feats = self.vgg(x)
        return [proj(feats[idx].mean(dim=(2, 3))) for proj, idx in zip(self.vgg_proj, self.vgg_stage_idx)]

    def forward(self, **kwargs):
        text_stages = self._text_stage_features(
            kwargs["content"], kwargs["content_masks"],
            kwargs["FTR_2"], kwargs["FTR_2_masks"],
            kwargs["FTR_3"], kwargs["FTR_3_masks"],
        )
        spatial_stages = self._spatial_stage_features(kwargs["img_spatial"])
        freq_stages = self._freq_stage_features(kwargs["img_freq_complex"])

        batch_size = kwargs["content"].size(0)
        state = self.init_state.expand(batch_size, -1).contiguous()
        for stage, t, s, f in zip(self.stages, text_stages, spatial_stages, freq_stages):
            state = stage(state, t, s, f)

        extra_embedding = self.extra_feature_mlp(kwargs["extra_features"])
        comment_embedding = self.comment_feature_mlp(kwargs["comment_features"])
        fused = torch.cat([state, extra_embedding, comment_embedding], dim=1)
        logit = self.classifier(fused).squeeze(1)
        return {"classify_pred": torch.sigmoid(logit), "fused_feature": fused}


class Trainer:
    def __init__(self, config):
        self.config = config
        self.save_path = os.path.join(config["param_dir"], "progressive_fusion")
        os.makedirs(self.save_path, exist_ok=True)

    def _loader(self, path, shuffle, train):
        cfg = self.config
        return get_multimodal_dataloader(
            path, cfg["image_root"], cfg["bert_path"], cfg["max_len"],
            cfg["extra_feature_dim"], cfg["comment_feature_dim"],
            cfg["batchsize"], shuffle, train=train,
        )

    @staticmethod
    def _to_device(batch, use_cuda):
        if not use_cuda:
            return batch
        return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train(self, logger=None):
        cfg = self.config
        self.model = ProgressiveFusionModel(cfg)
        if cfg["use_cuda"]:
            self.model = self.model.cuda()

        loss_fn = nn.BCELoss()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        recorder = Recorder(cfg["early_stop"])

        train_loader = self._loader(cfg["train_meta"], True, True)
        val_loader = self._loader(cfg["val_meta"], False, False)
        test_loader = self._loader(cfg["test_meta"], False, False)

        for epoch in range(cfg["epoch"]):
            print(f"---------- epoch {epoch} ----------")
            self.model.train()
            avg_loss = Averager()

            for batch in tqdm.tqdm(train_loader):
                batch = self._to_device(batch, cfg["use_cuda"])
                label = batch["label"]

                res = self.model(**batch)
                loss = loss_fn(res["classify_pred"], label.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss.add(loss.item())

            print("----- validating -----")
            val_results = self.evaluate(val_loader)
            print("val:", val_results)
            mark = recorder.add(val_results)
            if logger:
                logger.info(f"epoch {epoch} train_loss={avg_loss.item():.4f} val={val_results}")

            if mark == "save":
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "parameter_pfn.pkl"))
            if mark == "esc":
                break

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, "parameter_pfn.pkl")))
        test_results = self.evaluate(test_loader)
        print("test results:", test_results)
        return test_results, os.path.join(self.save_path, "parameter_pfn.pkl")

    def evaluate(self, dataloader):
        self.model.eval()
        pred, label = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader):
                batch = self._to_device(batch, self.config["use_cuda"])
                res = self.model(**batch)
                label.extend(batch["label"].detach().cpu().numpy().tolist())
                pred.extend(res["classify_pred"].detach().cpu().numpy().tolist())
        return metrics(label, pred)