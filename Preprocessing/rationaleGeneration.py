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

MESSAGE_TEMPLATE = 'Post title: "{content}"'

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
def makeRequest(client, content):
    userMessage = MESSAGE_TEMPLATE.format(content=content.replace('"', "'")[:500])
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
 
#Main function that gets the rationales with their respective dataset depending on which split we are talking about 
#(train, test, validate)
def processSplit(splitName, jsonPath):
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
 
        tdRationale, tdPred, csRationale, csPred = makeRequest(client, rec["content"])
        time.sleep(config.RATIONALE_SLEEP_BETWEEN_CALLS)

 
        recOut = dict(rec)
        recOut["td_rationale"] = tdRationale
        recOut["td_pred"] = tdPred
        recOut["td_acc"] = int(tdPred == rec["label"])
        recOut["cs_rationale"] = csRationale
        recOut["cs_pred"] = csPred
        recOut["cs_acc"] = int(csPred == rec["label"])
 
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
 
def processSplitTemp(splitName, jsonPath):
    print(f"\n==={splitName}===")
    #Load only the progress up until this point
    done = loadProgress(splitName)
    print(f"skipping LLM calls for now and using {len(done)}")
    #we put the done values inside a list to then put them in the final json
    finalRecords = list(done.values())
    
    outPath = os.path.join(config.RATIONALES, f"{'val' if splitName == 'validate' else splitName}.json")
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(finalRecords, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(finalRecords)} inside {outPath}")


def main():
    os.makedirs(config.RATIONALES, exist_ok=True)
    processSplitTemp("train", os.path.join(config.ARG_OUTPUT, "train_pre.json"))
    #processSplit("validate", os.path.join(config.ARG_OUTPUT, "val_pre.json"))
    #processSplit("test", os.path.join(config.ARG_OUTPUT, "test_pre.json"))
    print("\nStep 2 complete. You can now run textTraining.py")
 
if __name__ == "__main__":
    main()