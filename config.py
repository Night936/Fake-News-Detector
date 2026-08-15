"""
This files handles the model hyperparameters for BERT, the api calls and keys for the 
added LLM rationales and every path needed to find datasets and where to write the outputs
"""

#Required Imports
import os

#Paths for OneDrive to gather data
DRIVE_ROOT = "/content/drive/MyDrive/Eduardo - Fake News Project - MITACS 2026/Code" #Base OneDrive directory

# ---------------------------------------------------------------------------
# DATASET MODE SWITCH -- set to "mini" or "full".
# "mini"  -> Mini_Dataset/ (the ~3,700-row subset used for all development so far)
# "full"  -> Main_Dataset/ (the full Fakeddit dataset)
# Everything below (TRAIN_TSV, MINI_IMAGES, OUTPUT_DIR, etc.) is derived from
# this single switch, so flipping it here is the only change needed to move
# the whole pipeline from mini to full or back.
#
# IMPORTANT: the exact TSV/folder filenames under Main_Dataset have not been
# confirmed -- the mini dataset's files are prefixed "mini_" (e.g.
# mini_multimodal_train.tsv), but the full dataset's files may just be
# multimodal_train.tsv, or something else entirely. Run:
#   !ls "Main_Dataset/Metadata"
#   !ls "Main_Dataset/Comments"
# and correct DATASET_FILE_PREFIX / the individual filenames below to match
# what's actually there before running anything against "full".
# ---------------------------------------------------------------------------
DATASET_MODE = "full"  # switched from "mini" -- mini-dataset validation confirmed stable

if DATASET_MODE == "mini":
    ACTIVE_DATASET = os.path.join(DRIVE_ROOT, "Mini_Dataset")
    DATASET_FILE_PREFIX = "mini_"
elif DATASET_MODE == "full":
    ACTIVE_DATASET = os.path.join(DRIVE_ROOT, "Main_Dataset")
    DATASET_FILE_PREFIX = ""  # CONFIRM: full dataset files may not share the "mini_" prefix
else:
    raise ValueError(f"Unknown DATASET_MODE: {DATASET_MODE!r} -- expected 'mini' or 'full'")

# Back-compat alias -- some scripts/notebooks may still reference MINI_DATASET
# directly; keep it pointed at whichever dataset is actually active so those
# call sites don't need to change.
MINI_DATASET = ACTIVE_DATASET

#Metadata tsv files for training, testing and validating
TRAIN_TSV = os.path.join(ACTIVE_DATASET, "Metadata", f"{DATASET_FILE_PREFIX}multimodal_train.tsv")
TEST_TSV = os.path.join(ACTIVE_DATASET, "Metadata", f"{DATASET_FILE_PREFIX}multimodal_test.tsv")
VALIDATE_TSV = os.path.join(ACTIVE_DATASET, "Metadata", f"{DATASET_FILE_PREFIX}multimodal_validate.tsv")

#Image folder
MINI_IMAGES = os.path.join(ACTIVE_DATASET, "Images")

#Comments file
COMMENTS_TSV = os.path.join(ACTIVE_DATASET, "Comments", f"{DATASET_FILE_PREFIX}comments.tsv")

#Where to write outputs -- kept per-dataset so a "full" run's outputs never
#overwrite or mix with the "mini" run's outputs (e.g. Rationales/train.json
#means something very different at each scale).
OUTPUT_DIR = os.path.join(ACTIVE_DATASET, "Outputs")
ARG_OUTPUT = os.path.join(OUTPUT_DIR, "ARG")
RATIONALES = os.path.join(OUTPUT_DIR, "Rationales")
PARAMETERS = os.path.join(OUTPUT_DIR, "Parameters")
LOGS = os.path.join(OUTPUT_DIR, "Logs")

#Important labels to denote from the tsvs
MAIN_TEXT = "clean_title" #Basically the text that is going to be fed for analysis
SECONDARY_TEXT = "title" #The non clean text in case the first one cant be found
LABEL_SOURCE = "6_way_label" #The final dissemination on wether it is fake or not, since its 
                             #probabilistic it takes into account the 6 way label

# ---------------------------------------------------------------------------
# LLM Config -- DeepInfra (paid tier). Rationale generation cost scales
# directly with row count, which matters a lot more in "full" mode than it
# did in "mini" mode -- see MAX_ROWS_PER_SPLIT below.
# ---------------------------------------------------------------------------
GROQ_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # confirm exact string on DeepInfra's model page
GROQ_BASE_URL = "https://api.deepinfra.com/v1/openai"
RATIONALE_MAX_RETRIES = 3
RATIONALE_SLEEP_BETWEEN_CALLS = 0.05

# DeepInfra is a paid tier with real concurrent capacity (unlike Groq's free
# tier, which is TPM-capped around 12k tokens/min regardless of worker
# count) -- safe to run meaningfully higher concurrency here.
RATIONALE_MAX_WORKERS = 80

# ---------------------------------------------------------------------------
# BUDGET SAFETY NET for rationale generation. At ~500 input / ~200 output
# tokens per row on Llama-3.3-70B-Instruct-Turbo ($0.10/$0.32 per M tokens),
# each row costs roughly $0.000114. With DATASET_MODE = "full" (~683k rows
# total across train/val/test), an UNCAPPED run could cost roughly $78 for
# the complete dataset -- fine if intentional, expensive if triggered by
# accident. Set per-split caps below; processSplitParallel in
# Preprocessing/rationaleGeneration.py truncates its per-split "todo" list to
# this many NEW rows per run (existing checkpointed rows are unaffected and
# don't re-spend). None = no cap (process every remaining row).
#
# Example for a ~$17 budget (leaves headroom under a $20 balance for
# retries): roughly 150,000 rows total, split ~80/10/10:
# MAX_ROWS_PER_SPLIT = {"train": 120000, "validate": 15000, "test": 15000}
#
# ACTUAL VALUES IN USE: capped well below the budget-driven number above,
# because wall-clock time on a single free-tier Colab T4 session -- not
# money, not disk, not GPU RAM -- is the real constraint. Measured
# throughput on the mini dataset was ~10.4 rows/sec at batch_size=8, so
# 150,000 rows would be ~4 hours/epoch (30-40+ hours total across early
# stopping) -- incompatible with Colab's session limits (a disconnect was
# already hit training on just the ~3,700-row mini dataset). 20,000 training
# rows is ~7x the mini dataset (a real scale-up test) while keeping a full
# run inside a single session (~35 min/epoch at the same throughput).
MAX_ROWS_PER_SPLIT = {
    "train": 8000,
    "validate": 1000,
    "test": 1000,
}
# Set to None (i.e. MAX_ROWS_PER_SPLIT = None) to disable the cap entirely
# and process every remaining row in a split -- do this only once you're
# deliberately committing to the full remaining cost.

#BERT Hyperparameters
BERT_PATH = "bert-base-uncased"  
MAX_LEN = 64          #Since the Fakeddit ones are short, for full articles use 170
BATCH_SIZE = 32
EPOCHS = 15
EARLY_STOP = 4
LR = 5e-5
WEIGHT_DECAY = 5e-5
SEED = 3759
 
EMB_DIM = 768
CO_ATTENTION_DIM = 300
MLP_DIMS = [384]
MLP_DROPOUT = 0.2
 
RATIONALE_USEFULNESS_EVALUATOR_WEIGHT = 1.5
LLM_JUDGMENT_PREDICTOR_WEIGHT = 1.0
 
#Extra feature vector size: VADER (4) + lexical stylistic (8) = 12 (so, besides ARG)
#See Preprocessing/build_arg_format_dataset.py for the exact feature list.
EXTRA_FEATURE_DIM = 12
EXTRA_FEATURE_MLP_DIM = 64

#Social / comment-engagement feature vector size: 8, produced by
#Utils/commentFeatures.py -> extractCommentSectionFeatures():
#[log1p(comment_count), avg_compound, spread, avg_ups, top_level_frac,
# downvoted_frac, min_compound, max_compound]
#This is the "social branch" signal (comment volume/sentiment/engagement),
#separate from the per-post VADER/lexical EXTRA_FEATURE_DIM above.
COMMENT_FEATURE_DIM = 8
COMMENT_FEATURE_MLP_DIM = 32
 
USE_CUDA = True

# ---------------------------------------------------------------------------
# Progressive Fusion Network (Models/progressiveFusion.py) settings
# ---------------------------------------------------------------------------
PFN_FUSION_DIM = EMB_DIM
 
# NOTE for "full" mode: this was sized for the mini dataset's multimodal
# overlap (a few hundred fully-rationale-and-image-complete rows at a time).
# At full scale, if GPU memory allows, consider raising this -- but confirm
# actual available rows first (rationale coverage will lag behind the raw
# row count given MAX_ROWS_PER_SPLIT above, likely for some time).
PFN_BATCH_SIZE = 8
PFN_LR = 1e-4