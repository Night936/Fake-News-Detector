"""
Preprocessing/subsampleDataset.py

Trims the full-dataset preprocessing outputs (train_pre.json/train_images.json
and their val/test equivalents) down to a stratified, label-balanced sample
matching config.MAX_ROWS_PER_SPLIT -- BEFORE the expensive downstream steps
(verifyImages.py decoding every image, commentAggregation.py running VADER
sentiment over the full comments.tsv) run on rows that would never actually
reach training anyway.

WHY THIS IS NEEDED: MAX_ROWS_PER_SPLIT already caps rationaleGeneration.py
(it only processes that many NEW rows per run), but nothing upstream of that
knew about the cap -- verifyImages.py and commentAggregation.py were
processing all ~511,150 full-dataset rows regardless, ~20x more work than
what ever reaches training. This script closes that gap.

Run AFTER imageMetadataCreation.py (needs train_images.json etc. to exist)
and BEFORE verifyImages.py:

    !python Preprocessing/jsonCreation.py
    !python Preprocessing/imageMetadataCreation.py
    !python Preprocessing/subsampleDataset.py     # <-- NEW, goes here
    !python Preprocessing/verifyImages.py
    !python Preprocessing/commentAggregation.py
    !python Preprocessing/reuseMiniRationales.py
    !python -u Preprocessing/rationaleGeneration.py
    !python -u ProgressiveFusionTraining.py

Sampling is stratified by label (real/fake) to preserve each split's
original class ratio, and uses config.SEED for reproducibility. Original
full-size files are backed up with a .full.json suffix (never overwritten),
so re-running jsonCreation.py/imageMetadataCreation.py from scratch and
re-subsampling is always possible without re-downloading anything.

Safe to run multiple times: if the .full.json backups already exist, it
re-samples FROM those backups (not from an already-trimmed file), so running
this script twice doesn't compound down to an ever-smaller dataset.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def stratified_sample(records, target_n, rng):
    """Samples target_n records from `records`, preserving the original
    real/fake ratio as closely as integer counts allow. If target_n exceeds
    the available count for a label, takes everything available for that
    label (doesn't error, just returns fewer than target_n total in that
    edge case)."""
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


def subsample_split(split_name, cap_key, rng):
    """Subsamples one split's train_images.json (the sampling universe --
    narrower than train_pre.json since it's already filtered to rows with a
    verified-present image file) down to config.MAX_ROWS_PER_SPLIT[cap_key],
    then filters the corresponding train_pre.json to the same set of
    fakeddit_ids so both files stay consistent regardless of which one a
    downstream script reads from."""
    cap = config.MAX_ROWS_PER_SPLIT.get(cap_key) if config.MAX_ROWS_PER_SPLIT else None
    if cap is None:
        print(f"  {split_name}: no cap set for '{cap_key}' in MAX_ROWS_PER_SPLIT -- skipping (using full size)")
        return

    images_path = os.path.join(config.ARG_OUTPUT, f"{split_name}_images.json")
    pre_path = os.path.join(config.ARG_OUTPUT, f"{split_name}_pre.json")
    images_backup = images_path.replace(".json", ".full.json")
    pre_backup = pre_path.replace(".json", ".full.json")

    if not os.path.exists(images_path):
        print(f"  {split_name}: {images_path} not found -- run imageMetadataCreation.py first")
        return

    # If a .full.json backup already exists, this script has run before --
    # re-sample FROM the backup (the true full-size data), not from an
    # already-trimmed file, so repeated runs don't compound the sample down.
    source_images_path = images_backup if os.path.exists(images_backup) else images_path
    source_pre_path = pre_backup if os.path.exists(pre_backup) else pre_path

    with open(source_images_path, "r", encoding="utf-8") as f:
        full_images = json.load(f)
    with open(source_pre_path, "r", encoding="utf-8") as f:
        full_pre = json.load(f)

    if len(full_images) <= cap:
        print(f"  {split_name}: {len(full_images)} rows already <= cap ({cap}) -- no subsampling needed")
        return

    sampled_images = stratified_sample(full_images, cap, rng)
    sampled_ids = {rec["fakeddit_id"] for rec in sampled_images}
    sampled_pre = [rec for rec in full_pre if rec.get("fakeddit_id") in sampled_ids]

    # back up the true full-size files, ONLY if not already backed up
    if not os.path.exists(images_backup):
        os.rename(images_path, images_backup)
    if not os.path.exists(pre_backup):
        os.rename(pre_path, pre_backup)

    with open(images_path, "w", encoding="utf-8") as f:
        json.dump(sampled_images, f, ensure_ascii=False, indent=2)
    with open(pre_path, "w", encoding="utf-8") as f:
        json.dump(sampled_pre, f, ensure_ascii=False, indent=2)

    n_real = sum(1 for r in sampled_images if r["label"] == "real")
    n_fake = sum(1 for r in sampled_images if r["label"] == "fake")
    print(f"  {split_name}: {len(full_images)} -> {len(sampled_images)} rows "
          f"(real={n_real}, fake={n_fake}); full backup at {os.path.basename(images_backup)}")


def main():
    if config.DATASET_MODE != "full":
        print("WARNING: DATASET_MODE is not 'full' -- subsampling the mini "
              "dataset further is almost certainly not what you want. "
              "Continuing anyway in case that's intentional.")

    rng = random.Random(config.SEED)

    print("Subsampling full-dataset preprocessing outputs down to MAX_ROWS_PER_SPLIT...")
    subsample_split("train", "train", rng)
    subsample_split("val", "validate", rng)
    subsample_split("test", "test", rng)
    print("\nDone. verifyImages.py and commentAggregation.py will now only "
          "process the sampled rows.")


if __name__ == "__main__":
    main()