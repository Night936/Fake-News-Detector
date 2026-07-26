import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from Utils.commentFeatures import extractCommentSectionFeatures

TOP_N_FOR_LLM = 4
MAX_CHARS_PER_COMMENT = 120
MAX_COMMENTS_FOR_FEATURES = 50


def loadIdMap():
    path = os.path.join(config.ARG_OUTPUT, "source_id_map.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["map"]  # fakeddit_id (str) -> source_id (int)


def buildCommentDigest(comments_sorted_by_ups):
    top = comments_sorted_by_ups[:TOP_N_FOR_LLM]
    if not top:
        return "No comments available."
    lines = []
    for c in top:
        body = str(c["body"])[:MAX_CHARS_PER_COMMENT].replace("\n", " ")
        lines.append(f'- "{body}" ({c["ups"]} upvotes)')
    return "\n".join(lines)


def main():
    id_map = loadIdMap()

    comments_df = pd.read_csv(config.COMMENTS_TSV, sep="\t")
    comments_df = comments_df.dropna(subset=["body"])
    comments_df["submission_id"] = comments_df["submission_id"].astype(str)

    grouped = comments_df.groupby("submission_id")

    records = []
    for fakeddit_id, group in grouped:
        if fakeddit_id not in id_map:
            continue  # comment thread for a post that isn't in our mini dataset

        rows = group.to_dict("records")
        rows_sorted = sorted(rows, key=lambda r: r.get("ups", 0), reverse=True)

        bodies = [str(r["body"]) for r in rows_sorted[:MAX_COMMENTS_FOR_FEATURES]]
        ups = [r.get("ups", 0) for r in rows_sorted[:MAX_COMMENTS_FOR_FEATURES]]
        top_level = [bool(r.get("isTopLevel", True)) for r in rows_sorted[:MAX_COMMENTS_FOR_FEATURES]]

        records.append({
            "source_id": id_map[fakeddit_id],
            "fakeddit_id": fakeddit_id,
            "comment_count": len(rows),
            "comment_features": extractCommentSectionFeatures(bodies, ups, top_level),
            "comment_digest": buildCommentDigest(rows_sorted),
        })

    os.makedirs(config.ARG_OUTPUT, exist_ok=True)
    out_path = os.path.join(config.ARG_OUTPUT, "comments_by_post.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path} ({len(records)} posts with comments)")


if __name__ == "__main__":
    main()