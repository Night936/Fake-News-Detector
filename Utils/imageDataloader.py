#Required imports
import json
import os
import torch
import sys
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from transformers import BertTokenizer 

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Preprocessing.imagePreprocessing import (
    getFrequencyTransform,
    getSpatialTransform,
    getDFT,
)

LABELS = {"real": 0, "fake": 1}

PREDICTED_VALS = {"real": 0, "fake": 1, "other": 2}

FAILED_RATIONALE_MESSAGE = "Could not determine from available information"

class FakedditImageDataset(Dataset):
    def __init__(self, jsonPath, imageDir, train = True, requireImage = True):
        #We get the raw image
        raw = json.load(open(jsonPath, "r", encoding="utf-8"))

        #And pre process it 
        self.imageDir = imageDir
        self.spatialTransform = getSpatialTransform(train=train)
        self.frequencyTransform = getFrequencyTransform(train=train)

        if requireImage:
            self.records = []
            nMissing = 0
            for rec in raw:
                path = os.path.join(imageDir, rec["fakeddit_id"] + ".jpg")
                if os.path.exists(path):
                    self.records.append(rec)
                else:
                    nMissing += 1
        else:
            self.records = raw
        
        print(f"[IMAGES]: {len(self.records)} usable rows)\n")
        print(f"[IMAGES]: {nMissing} skipped bc that image doesnt exist bro")

    #Function to get the amount of usable records
    def getUsableRecords(self):
        return len(self.records)
        
    def loadImageTensors(self, fakedditId):
        path = os.path.join(self.imageDir, fakedditId + ".jpg")
        try:
            img = Image.open(path).convert("RGB")
            imgSpatial = self.spatialTransform(img)
            imgFrequency = self.frequencyTransform(getDFT(img))
            return imgSpatial, imgFrequency, torch.tensor(1.0)
        except Exception as e:
            return (
                torch.zeros(3, 224, 224),
                torch.zeros(3, 224, 224),
                torch.tensor(0.0)
            )
            
    def getItem(self, idx):
        rec = self.records[idx]
        imgSpatial, imgFrequency, hasImage = self.loadImageTensors(rec["fakeddit_id"])
        label = LABELS[rec["label"]]
        sourceId = rec["source_id"]

        return(imgSpatial, imgFrequency, hasImage,
                torch.tensor(label, dtype=torch.long),
                torch.tensor(sourceId, dtype=torch.long))
        
def getImageDataloader(jsonPath, imageDir, batchSize, shuffle, train=True):
    dataset = FakedditImageDataset(jsonPath, imageDir, train=train, requireImage=True)

    return DataLoader(
        dataset,
        batch_size = batchSize,
        shuffle = shuffle,
        num_workers = 2,
        pin_memory = True,
        drop_last = train,
    ) 

class FakedditMultimodalDataset(Dataset):
    def __init__(self, jsonPath, imageDir, bertPath, maxLen,
                 extra_feature_dim, train=True):
        raw = json.load(open(jsonPath, "r", encoding="utf-8"))
        self.tokenizer = BertTokenizer.from_pretrained(bertPath)
        self.maxLen = maxLen
        self.extra_feature_dim = extra_feature_dim
        self.imageDir = imageDir
        self.spatialTransform = getSpatialTransform(train=train)
        self.frequencyTransform = getFrequencyTransform(train=train)
 
        n_no_rationale = 0
        n_no_image     = 0
        self.records   = []
        for rec in raw:
            # Skip rows without real rationales
            if rec.get("td_rationale", "").startswith(FAILED_RATIONALE_MESSAGE):
                n_no_rationale += 1
                continue
            # Skip rows without images
            path = os.path.join(imageDir, rec["fakeddit_id"] + ".jpg")
            if not os.path.exists(path):
                n_no_image += 1
                continue
            self.records.append(rec)
 
        print(f"  FakedditMultimodalDataset: {len(self.records)} usable rows "
              f"({n_no_rationale} no rationale, {n_no_image} no image)")
 
    def _tokenize(self, text):
        ids = self.tokenizer.encode(
            text, max_length=self.max_len,
            add_special_tokens=True, padding="max_length", truncation=True,
        )
        ids    = torch.tensor(ids)
        masks  = (ids != self.tokenizer.pad_token_id).long()
        return ids, masks
 
    def _load_image_tensors(self, fakeddit_id):
        path = os.path.join(self.image_dir, fakeddit_id + ".jpg")
        try:
            img         = Image.open(path).convert("RGB")
            img_spatial = self.spatialTransform(img)
            img_freq    = self.frequencyTransform(getDFT(img))
            return img_spatial, img_freq, torch.tensor(1.0)
        except Exception:
            return (torch.zeros(3, 224, 224),
                    torch.zeros(3, 224, 224),
                    torch.tensor(0.0))
 
    def __len__(self):
        return len(self.records)
 
    def __getitem__(self, idx):
        rec = self.records[idx]
 
        content, content_masks = self._tokenize(rec["content"])
        ftr2, ftr2_masks       = self._tokenize(rec["td_rationale"])
        ftr3, ftr3_masks       = self._tokenize(rec["cs_rationale"])
 
        extra = torch.tensor(
            rec.get("extra_features", [0.0] * self.extra_feature_dim),
            dtype=torch.float,
        )
 
        img_spatial, img_freq, has_image = self._load_image_tensors(rec["fakeddit_id"])
 
        return {
            "content":        content,
            "content_masks":  content_masks,
            "FTR_2":          ftr2,
            "FTR_2_masks":    ftr2_masks,
            "FTR_2_pred":     torch.tensor(PREDICTED_VALS[rec["td_pred"]], dtype=torch.long),
            "FTR_2_acc":      torch.tensor(rec["td_acc"],         dtype=torch.long),
            "FTR_3":          ftr3,
            "FTR_3_masks":    ftr3_masks,
            "FTR_3_pred":     torch.tensor(PREDICTED_VALS[rec["cs_pred"]], dtype=torch.long),
            "FTR_3_acc":      torch.tensor(rec["cs_acc"],         dtype=torch.long),
            "extra_features": extra,
            "img_spatial":    img_spatial,
            "img_freq":       img_freq,
            "has_image":      has_image,
            "label":          torch.tensor(LABELS[rec["label"]], dtype=torch.long),
            "source_id":      torch.tensor(rec["source_id"],     dtype=torch.long),
        }
 
 
def get_multimodal_dataloader(json_path, image_dir, bert_path,
                               max_len, extra_feature_dim,
                               batch_size, shuffle, train=True):
    dataset = FakedditMultimodalDataset(
        json_path, image_dir, bert_path,
        max_len, extra_feature_dim, train=train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
        drop_last=train,
    )
 