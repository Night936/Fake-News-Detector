import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm

import config
from Models.imageModel import ImageExpertsModel
from Utils.imageDataloader import getImageDataloader # Fixed import 
from Utils.argUtils import Averager, Recorder, metrics


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ImageTrainer:
    def __init__(self, cfg):
        self.config = cfg
        self.save_path = os.path.join(cfg["param_dir"], "image_branch")
        os.makedirs(self.save_path, exist_ok=True)

    def train(self, logger=None):
        cfg = self.config
        self.model = ImageExpertsModel(cfg)
        if cfg["use_cuda"]:
            self.model = self.model.cuda()

        # FIXED: CrossEntropyLoss is required for a 2-neuron logit output
        loss_fn = nn.CrossEntropyLoss()
        
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        recorder = Recorder(cfg["early_stop"])

        # FIXED: Function name matches the import
        train_loader = getImageDataloader(cfg["train_meta"], cfg["image_root"], cfg["batchsize"], shuffle=True, train=True)
        val_loader   = getImageDataloader(cfg["val_meta"], cfg["image_root"], cfg["batchsize"], shuffle=False, train=False)
        test_loader  = getImageDataloader(cfg["test_meta"], cfg["image_root"], cfg["batchsize"], shuffle=False, train=False)

        for epoch in range(cfg["epoch"]):
            print(f"---------- epoch {epoch} ----------")
            self.model.train()
            avg_loss = Averager()

            # FIXED: Unpack all 5 variables returned by your dataloader
            for imgSpatial, imgFrequency, hasImage, labels, sourceIds in tqdm.tqdm(train_loader):
                
                if cfg["use_cuda"]:
                    imgSpatial = imgSpatial.cuda()
                    imgFrequency = imgFrequency.cuda()
                    labels = labels.cuda()

                # FIXED: Pass both image tensors to the Model
                logits = self.model(imgSpatial, imgFrequency)
                
                # FIXED: Calculate loss directly on the logits
                loss = loss_fn(logits, labels)

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
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "parameter_image.pkl"))
            if mark == "esc":
                break

        self.model.load_state_dict(torch.load(os.path.join(self.save_path, "parameter_image.pkl")))
        test_results = self.evaluate(test_loader)
        print("test results:", test_results)
        return test_results, os.path.join(self.save_path, "parameter_image.pkl")

    def evaluate(self, dataloader):
        self.model.eval()
        pred, label = [], []
        with torch.no_grad():
            # FIXED: Unpack all 5 variables here as well
            for imgSpatial, imgFrequency, hasImage, labels, sourceIds in tqdm.tqdm(dataloader):
                if self.config["use_cuda"]:
                    imgSpatial = imgSpatial.cuda()
                    imgFrequency = imgFrequency.cuda()
                    labels = labels.cuda()
                
                logits = self.model(imgSpatial, imgFrequency)
                
                # FIXED: Convert logits to class predictions (0 or 1) using argmax
                predictions = torch.argmax(logits, dim=1)
                
                label.extend(labels.detach().cpu().numpy().tolist())
                pred.extend(predictions.detach().cpu().numpy().tolist())
                
        return metrics(label, pred)


def main():
    setup_seed(config.SEED)
    os.makedirs(config.PARAMETERS, exist_ok=True)
    os.makedirs(config.LOGS, exist_ok=True)

    cfg = {
        "use_cuda": config.USE_CUDA and torch.cuda.is_available(),
        "lr": 5e-5,
        "weight_decay": config.WEIGHT_DECAY,
        "batchsize": 32,
        "epoch": config.EPOCHS,
        "early_stop": config.EARLY_STOP,
        "emb_dim": config.EMB_DIM,
        "param_dir": config.PARAMETERS,
        "train_meta": os.path.join(config.ARG_OUTPUT, "train_images.json"),
        "val_meta": os.path.join(config.ARG_OUTPUT, "val_images.json"),
        "test_meta": os.path.join(config.ARG_OUTPUT, "test_images.json"),
        "image_root": config.MINI_IMAGES,
        "model": {
            "mlp": {"dims": config.MLP_DIMS, "dropout": config.MLP_DROPOUT},
        },
    }

    if not cfg["use_cuda"]:
        print("WARNING: no GPU detected. In Colab: Runtime -> Change runtime type -> GPU.")

    trainer = ImageTrainer(cfg)
    test_results, model_path = trainer.train()

    results_path = os.path.join(config.LOGS, "image_branch_test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\nSaved model to: {model_path}")
    print(f"Saved test results to: {results_path}")


if __name__ == "__main__":
    main()