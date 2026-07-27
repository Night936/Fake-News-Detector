"""
Entry point for training the progressive multimodal fusion network (PFN),
replacing the separate text-branch / image-branch training with a single
end-to-end model that fuses both at four levels of depth (see
Models/progressiveFusion.py for the architecture).

Run from Colab, after the full preprocessing pipeline (in this order):

    !python Preprocessing/jsonCreation.py
    !python Preprocessing/imageMetadataCreation.py
    !python Preprocessing/verifyImages.py
    !python Preprocessing/commentAggregation.py
    !python Preprocessing/rationaleGeneration.py
    !python progressiveFusionTraining.py

FIX vs. the first draft of this file: train_meta/val_meta/test_meta used to
point at config.ARG_OUTPUT/{train,val,test}_images.json -- those files are
written by imageMetadataCreation.py and never carry td_rationale/cs_rationale/
comment_features, so FakedditMultimodalDataset's rationale filter would have
skipped every single row (it checks rec.get("td_rationale", ...), which is
always absent there), and training would have started with a zero-row
dataset. The rationale-augmented files -- config.RATIONALES/{train,val,test}.json,
produced by rationaleGeneration.py -- already carry fakeddit_id, content,
both rationales, extra_features, and comment_features together, which is
exactly what FakedditMultimodalDataset needs, so this now points there
instead. No separate merge step is required.

Needs timm (already in requirements.txt) for the Swin Transformer and VGG19
(features_only) backbones; both download pretrained ImageNet weights on
first use, so make sure the environment has outbound network access the
first time this runs.
"""

import json
import os
import random

import numpy as np
import torch

import config
from Models.progressiveFusion import Trainer


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
        "lr": config.PFN_LR,
        "weight_decay": config.WEIGHT_DECAY,
        "batchsize": config.PFN_BATCH_SIZE,
        "max_len": config.MAX_LEN,
        "epoch": config.EPOCHS,
        "early_stop": config.EARLY_STOP,
        "bert_path": config.BERT_PATH,
        "emb_dim": config.EMB_DIM,
        "fusion_dim": config.PFN_FUSION_DIM,
        "extra_feature_dim": config.EXTRA_FEATURE_DIM,
        "extra_feature_mlp_dim": config.EXTRA_FEATURE_MLP_DIM,
        "comment_feature_dim": config.COMMENT_FEATURE_DIM,
        "comment_feature_mlp_dim": config.COMMENT_FEATURE_MLP_DIM,
        "param_dir": config.PARAMETERS,
        "image_root": config.MINI_IMAGES,
        # FIX: point at the rationale-augmented output, not the
        # rationale-free image metadata files -- see module docstring.
        "train_meta": os.path.join(config.RATIONALES, "train.json"),
        "val_meta": os.path.join(config.RATIONALES, "val.json"),
        "test_meta": os.path.join(config.RATIONALES, "test.json"),
        "model": {
            "mlp": {"dims": config.MLP_DIMS, "dropout": config.MLP_DROPOUT},
        },
    }

    if not cfg["use_cuda"]:
        print("WARNING: no GPU detected. In Colab: Runtime -> Change runtime type -> GPU.")
        print("The Swin + VGG19 + BERT stack is heavy; CPU training will be very slow.")

    for path in (cfg["train_meta"], cfg["val_meta"], cfg["test_meta"]):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run Preprocessing/rationaleGeneration.py "
                f"(after jsonCreation.py, imageMetadataCreation.py, verifyImages.py, "
                f"and commentAggregation.py) before this script."
            )

    print("config:", json.dumps({k: v for k, v in cfg.items() if k != "model"}, indent=2))

    trainer = Trainer(cfg)
    test_results, model_path, test_predictions = trainer.train()

    results_path = os.path.join(config.LOGS, "progressive_fusion_test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)

    # Per-row probability output (source_id, label, predicted_probability,
    # predicted_label, absolute_error, correct) -- this is what supports a
    # calibration table / example table in the paper, since test_results
    # above is only the aggregate AUC/F1/precision/recall/acc dict.
    predictions_path = os.path.join(config.LOGS, "progressive_fusion_predictions.json")
    with open(predictions_path, "w") as f:
        json.dump(test_predictions, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved test results to: {results_path}")
    print(f"Saved per-row test predictions to: {predictions_path}")


if __name__ == "__main__":
    main()