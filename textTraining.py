"""
Entry point for training the text branch (ARG + rationales + VADER/lexical).

Run from Colab, after steps 1 and 2 in preprocessing/ have produced
train.json / val.json / test.json:

    !python train_text_branch.py

Requires models/layers.py and utils/utils.py from your existing ARG repo
to be present unmodified (they're generic and don't need any Fakeddit-
specific changes).
"""

import json
import os
import random

import numpy as np
import torch

import config
from Models.argFakeddit import Trainer


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(config.SEED)
    os.makedirs(config.PARAMETERS, exist_ok=True)
    os.makedirs(config.LOGS, exist_ok=True)

    cfg = {
        "use_cuda": config.USE_CUDA and torch.cuda.is_available(),
        "lr": config.LR,
        "weight_decay": config.WEIGHT_DECAY,
        "batchsize": config.BATCH_SIZE,
        "max_len": config.MAX_LEN,
        "epoch": config.EPOCHS,
        "early_stop": config.EARLY_STOP,
        "bert_path": config.BERT_PATH,
        "emb_dim": config.EMB_DIM,
        "co_attention_dim": config.CO_ATTENTION_DIM,
        "extra_feature_dim": config.EXTRA_FEATURE_DIM,
        "extra_feature_mlp_dim": config.EXTRA_FEATURE_MLP_DIM,
        "param_dir": config.PARAMETERS,
        "train_path": os.path.join(config.RATIONALES, "train.json"),
        "val_path": os.path.join(config.RATIONALES, "val.json"),
        "test_path": os.path.join(config.RATIONALES, "test.json"),
        "model": {
            "mlp": {"dims": config.MLP_DIMS, "dropout": config.MLP_DROPOUT},
            "rationale_usefulness_evaluator_weight": config.RATIONALE_USEFULNESS_EVALUATOR_WEIGHT,
            "llm_judgment_predictor_weight": config.LLM_JUDGMENT_PREDICTOR_WEIGHT,
        },
    }

    if not cfg["use_cuda"]:
        print("WARNING: no GPU detected. In Colab: Runtime -> Change runtime type -> GPU.")

    print("config:", json.dumps({k: v for k, v in cfg.items() if k != "model"}, indent=2))

    trainer = Trainer(cfg)
    test_results, model_path = trainer.train()

    results_path = os.path.join(config.LOGS, "text_branch_test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved test results to: {results_path}")


if __name__ == "__main__":
    main()