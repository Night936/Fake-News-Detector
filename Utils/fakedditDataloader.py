"""
Extension of the original ARG utils/dataloader.py.

Differences from the original:
  1. Reads the extra "extra_features" field (VADER + lexical vector) and
     returns it as an additional tensor, so the model can fuse it in.
  2. Uses modern pandas (no df.append, which is removed in pandas>=2.0).
  3. source_id is already a plain sequential int from step 1, so no casting
     surprises with Fakeddit's alphanumeric Reddit ids.

Everything else (word2input, label_dict, label_dict_ftr_pred, the overall
shape of the dataset) is intentionally kept identical to ARG's original so
that models/arg_fakeddit.py stays a close, easy-to-audit extension of
models/arg.py rather than a rewrite.
"""

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

label_dict = {"real": 0, "fake": 1, 0: 0, 1: 1}
label_dict_ftr_pred = {"real": 0, "fake": 1, "other": 2, 0: 0, 1: 1, 2: 2}


def word2input(texts, max_len, tokenizer):
    token_ids = []
    for text in texts:
        token_ids.append(
            tokenizer.encode(
                text,
                max_length=max_len,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
            )
        )
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = tokens != mask_token_id
    return token_ids, masks


def get_dataloader(path, max_len, batch_size, shuffle, bert_path, extra_feature_dim):
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    data_list = json.load(open(path, "r", encoding="utf-8"))
    rows = []
    for item in data_list:
        rows.append(
            {
                "content": item["content"],
                "label": item["label"],
                "id": item["source_id"],
                "FTR_2": item["td_rationale"],
                "FTR_3": item["cs_rationale"],
                "FTR_2_pred": item["td_pred"],
                "FTR_3_pred": item["cs_pred"],
                "FTR_2_acc": item["td_acc"],
                "FTR_3_acc": item["cs_acc"],
                "extra_features": item.get("extra_features", [0.0] * extra_feature_dim),
            }
        )
    df_data = pd.DataFrame(rows)

    content = df_data["content"].to_numpy()
    label = torch.tensor(df_data["label"].apply(lambda c: label_dict[c]).astype(int).to_numpy())
    item_id = torch.tensor(df_data["id"].astype(int).to_numpy())

    FTR_2_pred = torch.tensor(
        df_data["FTR_2_pred"].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy()
    )
    FTR_3_pred = torch.tensor(
        df_data["FTR_3_pred"].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy()
    )
    FTR_2_acc = torch.tensor(df_data["FTR_2_acc"].astype(int).to_numpy())
    FTR_3_acc = torch.tensor(df_data["FTR_3_acc"].astype(int).to_numpy())

    FTR_2 = df_data["FTR_2"].to_numpy()
    FTR_3 = df_data["FTR_3"].to_numpy()

    content_token_ids, content_masks = word2input(content, max_len, tokenizer)
    FTR_2_token_ids, FTR_2_masks = word2input(FTR_2, max_len, tokenizer)
    FTR_3_token_ids, FTR_3_masks = word2input(FTR_3, max_len, tokenizer)

    extra_features = torch.tensor(
        df_data["extra_features"].tolist(), dtype=torch.float
    )

    dataset = TensorDataset(
        content_token_ids,
        content_masks,
        FTR_2_pred,
        FTR_2_acc,
        FTR_3_pred,
        FTR_3_acc,
        FTR_2_token_ids,
        FTR_2_masks,
        FTR_3_token_ids,
        FTR_3_masks,
        label,
        item_id,
        extra_features,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=False,
        shuffle=shuffle,
    )
    return dataloader


def data2gpu(batch, use_cuda):
    fields = [
        "content", "content_masks", "FTR_2_pred", "FTR_2_acc", "FTR_3_pred", "FTR_3_acc",
        "FTR_2", "FTR_2_masks", "FTR_3", "FTR_3_masks", "label", "id", "extra_features",
    ]
    batch_data = dict(zip(fields, batch))
    if use_cuda:
        batch_data = {k: v.cuda() for k, v in batch_data.items()}
    return batch_data