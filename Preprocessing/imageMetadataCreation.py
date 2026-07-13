import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from Preprocessing.jsonCreation import deriveLabel


#function to load the id map from the output directory
def loadIdMap():
    path = os.path.join(config.ARG_OUTPUT, "source_id_map.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["map"], data["next_id"]
    return {}, 0

def buildSplit(tsv_path, split_name, image_root, id_map, next_id):
    print(f"Reading {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")

    records = []
    missing = 0
    skipped_no_image_flag = 0

    for _, row in df.iterrows():
        if not bool(row.get("hasImage", False)):
            skipped_no_image_flag += 1
            continue

        fakeddit_id = str(row.get("id", ""))
        filename = f"{fakeddit_id}.jpg"
        if not os.path.exists(os.path.join(image_root, filename)):
            missing += 1
            continue

        try:
            label = deriveLabel(row)
        except ValueError:
            continue

        if fakeddit_id in id_map:
            source_id = id_map[fakeddit_id]
        else: 
            source_id = next_id
            id_map[fakeddit_id] = source_id
            next_id += 1

        records.append({
            "source_id": source_id,   
            "fakeddit_id": fakeddit_id,
            "label": label,
            "image_filename": filename,
        })

    print(f"  {split_name}: {len(records)} usable, {missing} missing on disk, "
          f"{skipped_no_image_flag} flagged hasImage=False")
    return records, next_id


def main():
    os.makedirs(config.ARG_OUTPUT, exist_ok=True)

    id_map, next_id = loadIdMap()
    for split_name, tsv_path, out_name in [
        ("train", config.TRAIN_TSV, "train_images.json"),
        ("validate", config.VALIDATE_TSV, "val_images.json"),
        ("test", config.TEST_TSV, "test_images.json"),
    ]:
        records, next_id = buildSplit(tsv_path, split_name, config.MINI_IMAGES, id_map, next_id)

        n_fake = sum(1 for r in records if r["label"] == "fake")
        n_real = len(records) - n_fake
        print(f"  label balance ({split_name}): real={n_real}, fake={n_fake}")

        out_path = os.path.join(config.ARG_OUTPUT, out_name)
        
        with open(os.path.join(config.ARG_OUTPUT, "source_id_map.json"), "w", encoding="utf-8") as f:
            json.dump({"next_id": next_id, "map": id_map}, f, ensure_ascii=False, indent=2)
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  wrote {out_path}\n")

    print("Image metadata build complete. You can now run imageTraining.py")


if __name__ == "__main__":
    main()
