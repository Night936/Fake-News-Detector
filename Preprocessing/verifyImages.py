import json, os, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def verifySplit(json_path, image_root):
    records = json.load(open(json_path, "r", encoding="utf-8"))
    good, bad = [], []
    for rec in records:
        path = os.path.join(image_root, rec["image_filename"])
        try:
            with Image.open(path) as img:
                img.convert("RGB").load()
            good.append(rec)
        except Exception as e:
            bad.append({**rec, "error": str(e)})
    return good, bad

def main():
    for split_name, fname in [("train", "train_images.json"),
                               ("validate", "val_images.json"),
                               ("test", "test_images.json")]:
        path = os.path.join(config.ARG_OUTPUT, fname)
        good, bad = verifySplit(path, config.MINI_IMAGES)
        print(f"{split_name}: {len(good)} valid, {len(bad)} corrupted")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(good, f, ensure_ascii=False, indent=2)

        if bad:
            bad_path = os.path.join(config.ARG_OUTPUT, f"{split_name}_corrupted.json")
            with open(bad_path, "w", encoding="utf-8") as f:
                json.dump(bad, f, ensure_ascii=False, indent=2)
            print(f"  wrote {bad_path} — inspect these")

if __name__ == "__main__":
    main()