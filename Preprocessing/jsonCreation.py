"""
This code translates the tsvs from the fakeddit dataset into their equivalent jsons to
feed the LLM in order to get its rationale, it writes the jsons into the folder given 
inside the config
"""
#Required imports
import os
import sys
import json
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from Utils.textFeatures import extractFeatures

def getText(row):
    val = row.get(config.MAIN_TEXT, None)
    if val is None or (isinstance(val, float)) or str(val).strip() == "":
        val = row.get(config.SECONDARY_TEXT, "")
    return str(val) if val is not None else ""

#Derives the label based on the one selected inside the config
def deriveLabel(row):
    if str(config.LABEL_SOURCE) in row:
        match str(config.LABEL_SOURCE):
            case "6_way_label" | "3_way_label":
                return "real" if int(row["6_way_label"]) == 0 else "fake"
            case "2_way_label":
                return "real" if int(row["2_way_label"]) == 1 else "fake"
    else:
        raise ValueError(
            f"LABEL_SOURCE='{config.LABEL_SOURCE}' column not found in this TSV. "
            f"Available columns: {list(row.index)}"
        )
    
#Reads the entire TSV and generates personalized ids for every record inside the original fakeddit
#in order to not deal with the whole messy alphanumeric string it uses, receives a startId in case 
#it is restarting an older process
def convertSplit(tsvPath, splitName, startId):
    print(f"Reading the tsv inside {tsvPath}")
    df = pd.read_csv(tsvPath, sep="\t")
    print(f"Found {len(df)} rows and {list(df.columns)} columns")

    records = []
    nextId = startId 
    skipped = 0

    for _, row in df.iterrows():
        text = getText(row)
        if not text.strip():
            skipped += 1
            continue
 
        try:
            label = deriveLabel(row)
        except ValueError as e:
            raise
 
        record = {
            "source_id": nextId,
            "fakeddit_id": str(row.get("id", "")),
            "content": text,
            "label": label,
            "time": int(row["created_utc"]) if "created_utc" in row and pd.notna(row["created_utc"]) else -1,
            "split": splitName,
            "extra_features": extractFeatures(text),
            "has_image": bool(row.get("hasImage", False)),
            "image_url": str(row.get("image_url", "")),
        }
        records.append(record)
        nextId += 1
 
    print(f"  -> {len(records)} usable rows ({skipped} skipped: empty text)")
    return records, nextId

#For every split to be done it grabs the text inside its according tsv, gets its label and builds
#a record with a propietary source id to not use the fakeddit ones
def main():
    os.makedirs(config.ARG_OUTPUT, exist_ok=True)
 
    next_id = 0
    for split_name, tsv_path, out_name in [
        ("train", config.TRAIN_TSV, "train_pre.json"),
        ("validate", config.VALIDATE_TSV, "val_pre.json"),
        ("test", config.TEST_TSV, "test_pre.json"),
    ]:
        records, next_id = convertSplit(tsv_path, split_name, next_id)
 
        n_fake = sum(1 for r in records if r["label"] == "fake")
        n_real = len(records) - n_fake
        print(f"label balance ({split_name}): real={n_real}, fake={n_fake}")
 
        out_path = os.path.join(config.ARG_OUTPUT, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"wrote {out_path}\n")
 
    print("Step 1 complete. Next: run preprocessing/generate_rationales.py")
 
 
if __name__ == "__main__":
    main()