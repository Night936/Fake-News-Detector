"""
Preprocessing/subsampleRationales.py

Trims ALREADY-COMPLETED Rationales/{train,val,test}.json down to a smaller
row count (config.MAX_ROWS_PER_SPLIT) -- WITHOUT touching rationale
generation, image verification, or comment aggregation again.

WHY THIS EXISTS: subsampleDataset.py subsamples BEFORE rationale generation
(from the raw, pre-rationale dataset). If you already spent real API calls
generating rationales for a larger sample and then want to train on a
SMALLER sample to fit a session's wall-clock budget, re-running
subsampleDataset.py + rationaleGeneration.py would draw a fresh random
sample that mostly doesn't overlap with what you already generated --
wasting the completed work and requiring new API calls for the new sample.

This script instead draws the smaller sample directly FROM the completed
train.json/val.json/test.json (which already have rationales, extra_features,
comment_features -- everything needed for training), so nothing needs to be
regenerated. Zero new API calls.

Run directly from a Drive-mounted or locally-staged Rationales folder:

    !python Preprocessing/subsampleRationales.py

Backs up the full completed files as train.full.json etc. before trimming
(never overwrites the backup on a second run), so re-running this with a
different MAX_ROWS_PER_SPLIT always samples from the true full set.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def stratified_sample(records, target_n, rng):
    by_label = {}
    for rec in records:
        by_label.setdefault(rec["label"], []).append(rec)

    total = len(records)
    sampled = []
    for label, group in by_label.items():
        share = len(group) / total
        n_for_label = min(len(group), round(target_n * share))
        sampled.extend(rng.sample(group, n_for_label))

    rng.shuffle(sampled)
    return sampled


def subsample_rationale_file(filename, cap_key, rng):
    cap = config.MAX_ROWS_PER_SPLIT.get(cap_key) if config.MAX_ROWS_PER_SPLIT else None
    if cap is None:
        print(f"  {filename}: no cap set for '{cap_key}' -- skipping (using full size)")
        return

    path = os.path.join(config.RATIONALES, filename)
    backup_path = path.replace(".json", ".full.json")

    if not os.path.exists(path):
        print(f"  {filename}: {path} not found -- skipping")
        return

    source_path = backup_path if os.path.exists(backup_path) else path
    with open(source_path, "r", encoding="utf-8") as f:
        full_records = json.load(f)

    if len(full_records) <= cap:
        print(f"  {filename}: {len(full_records)} rows already <= cap ({cap}) -- no trimming needed")
        return

    sampled = stratified_sample(full_records, cap, rng)

    if not os.path.exists(backup_path):
        os.rename(path, backup_path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    n_real = sum(1 for r in sampled if r["label"] == "real")
    n_fake = sum(1 for r in sampled if r["label"] == "fake")
    print(f"  {filename}: {len(full_records)} -> {len(sampled)} rows "
          f"(real={n_real}, fake={n_fake}); full backup at {os.path.basename(backup_path)}")


def main():
    rng = random.Random(config.SEED)
    print("Subsampling completed rationale files down to MAX_ROWS_PER_SPLIT (no new API calls)...")
    subsample_rationale_file("train.json", "train", rng)
    subsample_rationale_file("val.json", "validate", rng)
    subsample_rationale_file("test.json", "test", rng)
    print("\nDone. ProgressiveFusionTraining.py will now train on the smaller sample.")


if __name__ == "__main__":
    main()