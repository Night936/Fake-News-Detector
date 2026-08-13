"""
Preprocessing/reuseMiniRationales.py

Pre-seeds the FULL dataset's rationale checkpoint files with rationales
already generated for the MINI dataset, so rationaleGeneration.py doesn't
re-spend LLM calls on posts it's already processed.

WHY THIS ISN'T A STRAIGHT FILE COPY: Mini_Dataset/Outputs/ and
Main_Dataset/Outputs/ are separate folders (config.py's DATASET_MODE switch
points ACTIVE_DATASET at one or the other), and jsonCreation.py assigns a
fresh, dense source_id to every row EACH TIME IT RUNS -- so mini's
source_id=47 and full's source_id=47 are almost certainly different posts.
Matching on source_id would silently corrupt data.

The correct join key is fakeddit_id (the original Reddit submission id),
which IS stable across both datasets since the mini dataset is a stratified
subset of the full one.

Run this AFTER jsonCreation.py + imageMetadataCreation.py + verifyImages.py
+ commentAggregation.py have completed for the FULL dataset (so full-dataset
source_ids, extra_features, and comment_features already exist), and BEFORE
rationaleGeneration.py.

    !python Preprocessing/reuseMiniRationales.py

Safe to run multiple times -- it only ever ADDS to the full dataset's
checkpoint files, never removes/overwrites, and skips fakeddit_ids that are
already present in the full checkpoint.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

FAILED_RATIONALE_MESSAGE = "Could not determine from available information"


def load_mini_rationale_lookup():
    """Builds one fakeddit_id -> rationale-fields dict from ALL THREE mini
    splits combined. A post that was in the mini dataset's train split could
    land in the full dataset's val or test split (the full dataset's own
    TSVs define the authoritative split), so we don't assume split names
    line up -- we just build one big lookup and match wherever a full-split
    row's fakeddit_id happens to hit."""
    mini_root = os.path.join(config.DRIVE_ROOT, "Mini_Dataset", "Outputs", "Rationales")
    lookup = {}
    n_skipped_bad = 0

    for split_file in ("train.json", "val.json", "test.json"):
        path = os.path.join(mini_root, split_file)
        if not os.path.exists(path):
            print(f"  (skipping {split_file} -- not found at {path})")
            continue
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            # Don't reuse rows that failed generation on the mini side --
            # those should get a fresh attempt on the full-dataset run
            # rather than propagating the same failure forward.
            if rec.get("td_rationale", "").startswith(FAILED_RATIONALE_MESSAGE):
                n_skipped_bad += 1
                continue
            fid = rec.get("fakeddit_id")
            if fid is None:
                continue
            lookup[fid] = {
                "td_rationale": rec["td_rationale"],
                "td_pred": rec["td_pred"],
                "cs_rationale": rec["cs_rationale"],
                "cs_pred": rec["cs_pred"],
            }
        print(f"  loaded {len(records)} rows from mini/{split_file}")

    print(f"Mini rationale lookup: {len(lookup)} usable fakeddit_ids "
          f"({n_skipped_bad} skipped -- were themselves failed generations)")
    return lookup


def seed_split(split_name, pre_json_path, comments_by_id, mini_lookup):
    """For one FULL-dataset split, cross-reference against the mini lookup
    and write a checkpoint line for every match -- using the FULL dataset's
    own source_id, label, extra_features (from pre_json_path) and
    comment_features (freshly computed for the full dataset's own comment
    threads, which may differ from the mini dataset's comment coverage),
    combined with the REUSED td/cs rationale fields from mini."""
    if not os.path.exists(pre_json_path):
        print(f"  {split_name}: {pre_json_path} not found, skipping")
        return 0

    with open(pre_json_path, "r", encoding="utf-8") as f:
        full_records = json.load(f)

    ckpt_path = os.path.join(config.RATIONALES, f"{split_name}_checkpoint.jsonl")
    os.makedirs(config.RATIONALES, exist_ok=True)

    # don't double-seed rows already in the checkpoint (from a previous
    # partial run, or a previous run of this script)
    already_done = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    already_done.add(json.loads(line)["source_id"])

    n_seeded = 0
    with open(ckpt_path, "a", encoding="utf-8") as ckpt_file:
        for rec in full_records:
            if rec["source_id"] in already_done:
                continue
            fid = rec.get("fakeddit_id")
            match = mini_lookup.get(fid)
            if match is None:
                continue  # not in the mini dataset -- will need real generation

            comment_info = comments_by_id.get(rec["source_id"], {})
            rec_out = dict(rec)
            rec_out["td_rationale"] = match["td_rationale"]
            rec_out["td_pred"] = match["td_pred"]
            rec_out["td_acc"] = int(match["td_pred"] == rec["label"])
            rec_out["cs_rationale"] = match["cs_rationale"]
            rec_out["cs_pred"] = match["cs_pred"]
            rec_out["cs_acc"] = int(match["cs_pred"] == rec["label"])
            rec_out["comment_features"] = comment_info.get(
                "comment_features", [0.0] * config.COMMENT_FEATURE_DIM
            )
            rec_out["comment_count"] = comment_info.get("comment_count", 0)

            ckpt_file.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n_seeded += 1

    print(f"  {split_name}: seeded {n_seeded} rows reused from mini dataset "
          f"(out of {len(full_records)} total)")
    return n_seeded


def main():
    if config.DATASET_MODE != "full":
        raise RuntimeError(
            "config.DATASET_MODE must be 'full' to run this migration -- "
            "it seeds the FULL dataset's checkpoint from the MINI dataset's "
            "already-completed rationales."
        )

    print("Loading mini-dataset rationale lookup...")
    mini_lookup = load_mini_rationale_lookup()

    comments_path = os.path.join(config.ARG_OUTPUT, "comments_by_post.json")
    comments_list = json.load(open(comments_path)) if os.path.exists(comments_path) else []
    comments_by_id = {c["source_id"]: c for c in comments_list}
    if not comments_list:
        print("WARNING: comments_by_post.json not found for the full dataset -- "
              "run Preprocessing/commentAggregation.py first.")

    total_seeded = 0
    total_seeded += seed_split("train", os.path.join(config.ARG_OUTPUT, "train_pre.json"), comments_by_id, mini_lookup)
    total_seeded += seed_split("validate", os.path.join(config.ARG_OUTPUT, "val_pre.json"), comments_by_id, mini_lookup)
    total_seeded += seed_split("test", os.path.join(config.ARG_OUTPUT, "test_pre.json"), comments_by_id, mini_lookup)

    print(f"\nTotal rows seeded from mini dataset: {total_seeded}")
    print("Run Preprocessing/rationaleGeneration.py next -- it will skip these "
          "automatically and only generate rationales for the remaining rows.")


if __name__ == "__main__":
    main()