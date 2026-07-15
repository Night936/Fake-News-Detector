# Preprocessing/imageTextAlignment.py
"""
Computes CLIP similarity between each post's text and its image, as a
preprocessing step (like extra_features) -- not trained end-to-end, so it's
usable immediately without touching the training loop.
"""
import json, os, sys
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    images_meta = json.load(open(os.path.join(config.ARG_OUTPUT, "train_images.json")))
    pre_meta = {r["fakeddit_id"]: r for r in json.load(open(os.path.join(config.ARG_OUTPUT, "train_pre.json")))}

    results = []
    with torch.no_grad():
        for rec in images_meta:
            content = pre_meta.get(rec["fakeddit_id"], {}).get("content")
            if content is None:
                continue
            img_path = os.path.join(config.MINI_IMAGES, rec["image_filename"])
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=[content], images=image, return_tensors="pt", padding=True, truncation=True).to(device)
            out = model(**inputs)
            sim = torch.cosine_similarity(out.image_embeds, out.text_embeds).item()
            results.append({"source_id": rec["source_id"], "image_text_similarity": sim})

    out_path = os.path.join(config.ARG_OUTPUT, "image_text_alignment.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"wrote {out_path} ({len(results)} pairs)")

if __name__ == "__main__":
    main()