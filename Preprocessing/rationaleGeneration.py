#Required Imports
import json
import os
import sys
import time 
from openai import OpenAI

#Allows this file to go 2 directories above to get the config file
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

PREDICTED_VALS = {"real", "fake", "other"}

#This prompt tells the LLM to analyze the post based on how its written, worded and presented
TEXTUAL_PROMPT = (
    "You are an expert at analyzing the writing style and presentation of social "
    "media posts to judge whether they describe real or fabricated/misleading "
    "content. Focus ONLY on textual clues: wording, tone, sensationalism, "
    "exaggeration, vagueness, clickbait patterns, internal inconsistency. "
    "Do not use outside world knowledge to judge truthfulness -- judge purely "
    "from how the text is written."
)

#This prompt tells the LLM to check for the veracity of the content, can it actually happen?
#based on commonsense
CONTENT_PROMPT = (
    "You are an expert fact-checker who judges social media post titles using "
    "general world/commonsense knowledge: is the claimed event plausible? Does "
    "it contradict well-known facts? Could this realistically happen? Focus on "
    "real-world plausibility rather than writing style."
)

#User message template that can be used with any content
MESSAGE_TEMPLATE = (
    'Post title: "{content}"\n\n'
    "Respond with ONLY a JSON object, no markdown fences, no extra text:\n"
    '{{"rationale": "<2-3 sentence rationale>", "prediction": "real" | "fake" | "other"}}'
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

def makeRequest(client , prompt, content):
    userMessage = MESSAGE_TEMPLATE.format(content=content.replace('"', "'")[:500])
    lastError = None

    for attempt in range(config.RATIONALE_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model = config.GROQ_MODEL,
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": userMessage},
                ],
                temperature = 0.2,
                max_tokens = 200,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
            parsed = json.loads(raw)
            pred = str(parsed.get("prediction", "other")).strip().lower()
            if pred not in PREDICTED_VALS:
                pred = "other"
            rationale = str(parsed.get("rationale", "")).strip()
            if not rationale:
                rationale = "No rationale provided."
            return rationale, pred
        except Exception as e:
            lastError = e
            time.sleep(1.0 + attempt)
    print(f"LLM call failed after {lastError}; using other")
    return "Could not determine from available information", "other"

def _checkpoint_path(split_name):
    return os.path.join(config.RATIONALES, f"{split_name}_checkpoint.jsonl")
 
 
def _load_checkpoint(split_name):
    done = {}
    path = _checkpoint_path(split_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done[rec["source_id"]] = rec
    return done
 
 
def process_split(split_name, pre_json_path):
    print(f"\n=== {split_name} ===")
    with open(pre_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
 
    done = _load_checkpoint(split_name)
    print(f"  {len(records)} total rows, {len(done)} already done (resuming)")
 
    client = getClient()
    ckpt_path = _checkpoint_path(split_name)
    ckpt_file = open(ckpt_path, "a", encoding="utf-8")
 
    for i, rec in enumerate(records):
        if rec["source_id"] in done:
            continue
 
        td_rationale, td_pred = makeRequest(client , TEXTUAL_PROMPT, rec["content"])
        time.sleep(config.RATIONALE_SLEEP_BETWEEN_CALLS)
        cs_rationale, cs_pred = makeRequest(client , CONTENT_PROMPT, rec["content"])
        time.sleep(config.RATIONALE_SLEEP_BETWEEN_CALLS)
 
        rec_out = dict(rec)
        rec_out["td_rationale"] = td_rationale
        rec_out["td_pred"] = td_pred
        rec_out["td_acc"] = int(td_pred == rec["label"])
        rec_out["cs_rationale"] = cs_rationale
        rec_out["cs_pred"] = cs_pred
        rec_out["cs_acc"] = int(cs_pred == rec["label"])
 
        ckpt_file.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
        ckpt_file.flush()
        done[rec["source_id"]] = rec_out
 
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)} done")
 
    ckpt_file.close()
 
    final_records = [done[r["source_id"]] for r in records]
    out_path = os.path.join(config.RATIONALES, f"{'val' if split_name == 'validate' else split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, ensure_ascii=False, indent=2)
    print(f"  wrote {out_path}")
 
 
def main():
    os.makedirs(config.RATIONALES, exist_ok=True)
    process_split("train", os.path.join(config.ARG_OUTPUT, "train_pre.json"))
    process_split("validate", os.path.join(config.ARG_OUTPUT, "val_pre.json"))
    process_split("test", os.path.join(config.ARG_OUTPUT, "test_pre.json"))
    print("\nStep 2 complete. You can now run train_text_branch.py")
 
 
if __name__ == "__main__":
    main()