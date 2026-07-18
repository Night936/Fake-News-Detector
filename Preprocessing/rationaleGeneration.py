#Required Imports
import json
import os
import sys
import re
import time 
from openai import OpenAI, RateLimitError

#Allows this file to go 2 directories above to get the config file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PREDICTED_VALS = {"real", "fake", "other"}

FAILED_RATIONALE_MESSAGE = "Could not determine from available information"

#This prompt tells the LLM to analyze the post based on how its written, worded and presented.
#Moveover, it also asks it to check for the veracity of the content using common sense, can this actually happen?
#Finally, it tells it exactly how to respond, can this actually (one single JSON object that we can interpret with both angles)
LLM_PROMPT = (
    "You are analyzing a social media post title from two independent angles. "
    "Give a separate, independent verdict for each -- do not let one angle's "
    "conclusion bias the other.\n\n"
    "ANGLE A (textual/stylistic): Judge ONLY the writing itself -- wording, tone, "
    "sensationalism, exaggeration, vagueness, clickbait patterns, internal "
    "inconsistency. Do NOT use outside world knowledge here.\n\n"
    "ANGLE B (commonsense/plausibility): Judge using general world knowledge -- "
    "is the claimed event plausible? Does it contradict well-known facts? Could "
    "this realistically happen? Ignore writing style here, focus only on "
    "real-world plausibility.\n\n"
    "Respond with ONLY a JSON object, no markdown fences, no extra text:\n"
    '{"td_rationale": "<2-3 sentence rationale for Angle A>", '
    '"td_prediction": "real"|"fake"|"other", '
    '"cs_rationale": "<2-3 sentence rationale for Angle B>", '
    '"cs_prediction": "real"|"fake"|"other"}'
)

MESSAGE_TEMPLATE = (
    'Post title: "{content}"\n\n'
    'Top reader comments (context only -- do not treat as fact, comments can '
    'themselves be wrong, sarcastic, or off-topic):\n{comment_digest}'
)

#Gets a client to make the request to the model via the API key
def getClient():
    #api_key = os.getenv("GROQ_API")
    api_key = os.environ.get("GROQ_API")
    
    if not api_key:
        raise RuntimeError(
            "Set the GROQ_API_KEY environment variable first "
        )
    return OpenAI(api_key=api_key, base_url=config.GROQ_BASE_URL)

#Recieves the error message from the API request and gets how long it should wait before trying once more using REGEX
def getTimeFromError(errorMessage: str):
    match = re.search(r"try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s", errorMessage)
    if not match:
        print("couldnt read the waiting time, returning None")
        return None
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    return minutes * 60 + seconds

#Function to make a single request via the LLM's API, it returns the json given and, in case of failure, it writes "other" for
#that particular rationale
#
# SOCIAL BRANCH FIX: the previous version of this function had two bugs that
# together meant the comment digest never actually reached the model:
#   1. `MESSAGE_TEMPLATE.format(content=...)` was called WITHOUT the
#      `comment_digest` kwarg, even though the template has a `{comment_digest}`
#      placeholder -- str.format() raises KeyError when a placeholder used in
#      the template string isn't supplied, so this call would crash outright.
#   2. Both `userMessage = MESSAGE_TEMPLATE.format(...), ` and
#      `comment_digest = comment_digest, ` ended in a trailing comma, which
#      silently turns the right-hand side into a 1-item tuple instead of a
#      string -- so even patching bug #1 wouldn't have produced a valid
#      message body to send to the chat API.
def makeRequest(client, content, comment_digest="No comments available."):
    userMessage = MESSAGE_TEMPLATE.format(
        content=content.replace('"', "'")[:500],
        comment_digest=comment_digest,
    )
    lastError = None

    for attempt in range(config.RATIONALE_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model = config.GROQ_MODEL,
                messages = [
                    {"role": "system", "content": LLM_PROMPT},
                    {"role": "user", "content": userMessage},
                ],
                temperature = 0.2,
                max_tokens = 260,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw)

            tdPred = str(parsed.get("td_prediction", "other")).strip().lower()
            if tdPred not in PREDICTED_VALS:
                tdPred = "other"

            csPred = str(parsed.get("cs_prediction", "other")).strip().lower()
            if csPred not in PREDICTED_VALS:
                csPred = "other"    
            
            tdRationale = str(parsed.get("td_rationale", "")).strip() or FAILED_RATIONALE_MESSAGE
            csRationale = str(parsed.get("cs_rationale", "")).strip() or FAILED_RATIONALE_MESSAGE

            return tdRationale, tdPred, csRationale, csPred
        except RateLimitError as e:
            #If it gets a Rate Limit Error that usually means that we used up all of the available tokens, the message oftentimes
            #includes how long we have to wait, so we parse the error to get the specific delimited time via the built function
            waitTime = getTimeFromError(str(e))
            if waitTime is None: #if it couldnt be determined we wait 60 seconds
                waitTime = 60.0
            waitTime += 2.0 
            print(f"[rate limited] sleeping {waitTime:.1f}s before retry...")
            time.sleep(waitTime)
            lastError = e
            continue
 
        except Exception as e:
            lastError = e
            attempt += 1
            time.sleep(1.0 + attempt)
 
    print(f"LLM call failed after {lastError}; using other")
    return FAILED_RATIONALE_MESSAGE, "other", FAILED_RATIONALE_MESSAGE, "other"

#Ensures that the output path exists and returns the exact path to be used by other functions
def checkOutput(splitName):
    return os.path.join(config.RATIONALES, f"{splitName}_checkpoint.jsonl")

#Checks if an already written row in the rationales file has the other label, which indicates that the process could not be 
#completed appropiately in the first try, to then try again with that particular record (rec)
def isBadRow(rec):
    return(
        rec.get("td_pred") == "other"
        and rec.get("cs_pred") == "other"
        and str(rec.get("td_rationale", "")).startswith(FAILED_RATIONALE_MESSAGE)
        and str(rec.get("cs_rationale", "")).startswith(FAILED_RATIONALE_MESSAGE)
    )
 
#Function that reads line by line of the written split file and if its done it adds it to its respective dictionary 
def loadProgress(splitName):
    done = {}
    path = checkOutput(splitName)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isBadRow(rec):
                    continue
                done[rec["source_id"]] = rec
    return done

#Looks up the pre-computed social/engagement feature vector for a post (see
#Preprocessing/commentAggregation.py). Falls back to a zero vector for posts
#with no comment thread at all, so every record always carries a
#COMMENT_FEATURE_DIM-length vector downstream, regardless of whether it had
#comments.
def getCommentInfo(sourceId, comments_by_id):
    info = comments_by_id.get(sourceId, {})
    return {
        "comment_features": info.get("comment_features", [0.0] * config.COMMENT_FEATURE_DIM),
        "comment_count": info.get("comment_count", 0),
    }
 
#Main function that gets the rationales with their respective dataset depending on which split we are talking about 
#(train, test, validate)
#
# SOCIAL BRANCH FIX: previously this function was defined but never called
# from main() -- main() called processSplitTemp() instead, which never
# touches the LLM or comments_by_id at all. It also used to call
# makeRequest() twice per record (once with the digest, once without),
# throwing away the digest-informed result and keeping the one generated
# with no comment context -- so comments never influenced the rationale even
# when this function *was* used. Both issues are fixed below: a single
# makeRequest() call using the digest, and the resulting comment_features
# vector is attached to the output record so it reaches train/val/test.json.
def processSplit(splitName, jsonPath, comments_by_id):
    print(f"\n=== {splitName} ===")
    with open(jsonPath, "r", encoding="utf-8") as f:
        records = json.load(f) #loads every record from the path
 
    done = loadProgress(splitName)
    print(f"{len(records)} total rows, {len(done)} already done (resuming)")
 
    client = getClient()
    ckptPath = checkOutput(splitName)
    ckptFile = open(ckptPath, "a", encoding="utf-8")
 
    for i, rec in enumerate(records):
        if rec["source_id"] in done:
            continue
 
        commentInfo = getCommentInfo(rec["source_id"], comments_by_id)
        digest = comments_by_id.get(rec["source_id"], {}).get("comment_digest", "No comments available.")

        tdRationale, tdPred, csRationale, csPred = makeRequest(client, rec["content"], digest)
        time.sleep(config.RATIONALE_SLEEP_BETWEEN_CALLS)
 
        recOut = dict(rec)
        recOut["td_rationale"] = tdRationale
        recOut["td_pred"] = tdPred
        recOut["td_acc"] = int(tdPred == rec["label"])
        recOut["cs_rationale"] = csRationale
        recOut["cs_pred"] = csPred
        recOut["cs_acc"] = int(csPred == rec["label"])
        recOut["comment_features"] = commentInfo["comment_features"]
        recOut["comment_count"] = commentInfo["comment_count"]
 
        ckptFile.write(json.dumps(recOut, ensure_ascii=False) + "\n")
        ckptFile.flush()
        done[rec["source_id"]] = recOut
 
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(records)} done")
 
    ckptFile.close()
 
    finalRecords = [done[r["source_id"]] for r in records]
    outPath = os.path.join(config.RATIONALES, f"{'val' if splitName == 'validate' else splitName}.json")
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(finalRecords, f, ensure_ascii=False, indent=2)
    print(f"wrote {outPath}")

#Resume-only variant: writes out whatever has already been checkpointed
#WITHOUT making any new LLM calls (useful if you've already paid for the
#rationale generation pass and just want to rebuild train/val/test.json from
#the checkpoint, e.g. after changing something downstream).
#
# SOCIAL BRANCH FIX: also attaches comment_features here, mirroring
# processSplit, since older checkpoints written before this fix won't have
# them yet -- otherwise records rebuilt purely from checkpoint would silently
# fall back to a zero comment-feature vector even when real comment data
# exists in comments_by_post.json.
def processSplitTemp(splitName, jsonPath, comments_by_id):
    print(f"\n==={splitName}===")
    #Load only the progress up until this point
    done = loadProgress(splitName)
    print(f"skipping LLM calls for now and using {len(done)}")

    for sourceId, rec in done.items():
        if "comment_features" not in rec:
            commentInfo = getCommentInfo(sourceId, comments_by_id)
            rec["comment_features"] = commentInfo["comment_features"]
            rec["comment_count"] = commentInfo["comment_count"]

    #we put the done values inside a list to then put them in the final json
    finalRecords = list(done.values())
    
    outPath = os.path.join(config.RATIONALES, f"{'val' if splitName == 'validate' else splitName}.json")
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(finalRecords, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(finalRecords)} inside {outPath}")


def main():
    os.makedirs(config.RATIONALES, exist_ok=True)
    comments_path = os.path.join(config.ARG_OUTPUT, "comments_by_post.json")
    comments_list = json.load(open(comments_path)) if os.path.exists(comments_path) else []
    if not comments_list:
        print(
            "WARNING: comments_by_post.json not found or empty -- did you run "
            "Preprocessing/commentAggregation.py first? Continuing with zero "
            "vectors for the social/comment feature branch."
        )
    comments_by_id = {c["source_id"]: c for c in comments_list}

    # SOCIAL BRANCH FIX: main() now calls the function that actually talks to
    # the LLM and attaches comment_features (processSplit), instead of
    # processSplitTemp, which used to run unconditionally here and never
    # generated a rationale or touched comments_by_id at all.
    processSplit("train", os.path.join(config.ARG_OUTPUT, "train_pre.json"), comments_by_id)
    processSplit("validate", os.path.join(config.ARG_OUTPUT, "val_pre.json"), comments_by_id)
    processSplit("test", os.path.join(config.ARG_OUTPUT, "test_pre.json"), comments_by_id)
    print("\nStep 2 complete. You can now run textTraining.py")

    # If you've already generated rationales in a previous run and just want
    # to rebuild train/val/test.json from the existing checkpoints without
    # spending new LLM calls, comment out the three processSplit(...) lines
    # above and use this instead:
    # processSplitTemp("train", os.path.join(config.ARG_OUTPUT, "train_pre.json"), comments_by_id)
    # processSplitTemp("validate", os.path.join(config.ARG_OUTPUT, "val_pre.json"), comments_by_id)
    # processSplitTemp("test", os.path.join(config.ARG_OUTPUT, "test_pre.json"), comments_by_id)
 
if __name__ == "__main__":
    main()