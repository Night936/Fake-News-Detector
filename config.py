"""
This files handles the model hyperparameters for BERT, the api calls and keys for the 
added LLM rationales and every path needed to find datasets and where to write the outputs
"""

#Required Imports
import os

#Paths for OneDrive to gather data
#DRIVE_ROOT = os.getenv("DRIVE_ROOT") 
DRIVE_ROOT = "/content/drive/MyDrive/Eduardo - Fake News Project - MITACS 2026/Code" #Base OneDrive directory
MINI_DATASET = "/content/drive/MyDrive/Eduardo - Fake News Project - MITACS 2026/Code/Mini_Dataset"
#MINI_DATASET = os.getenv("MINI_DATASET")

#Metadata tsv files for training, testing and validating
TRAIN_TSV = os.path.join(MINI_DATASET, "Metadata", "mini_multimodal_train.tsv")
TEST_TSV = os.path.join(MINI_DATASET, "Metadata", "mini_multimodal_test.tsv")
VALIDATE_TSV = os.path.join(MINI_DATASET, "Metadata", "mini_multimodal_validate.tsv")

#Image folder
MINI_IMAGES = os.path.join(MINI_DATASET, "Images")

#Comments file
COMMENTS_TSV = os.path.join(MINI_DATASET, "Comments", "mini_comments.tsv")

#Where to write outputs
OUTPUT_DIR = os.path.join(MINI_DATASET, "Outputs")
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
# LLM Config -- REVERTED to Groq for tonight's run (temporary/preliminary
# results for the paper). DeepInfra requires account approval that hasn't
# come through yet; Groq's free tier is available right now. Swapping back
# to DeepInfra (or Claude) later is just re-pointing GROQ_MODEL/
# GROQ_BASE_URL and getClient() in Preprocessing/rationaleGeneration.py --
# nothing else in the pipeline needs to change.
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
RATIONALE_MAX_RETRIES = 3
RATIONALE_SLEEP_BETWEEN_CALLS = 0.05  #seconds -- kept small; free-tier throttling is handled
                                       #by the 429 retry-after backoff, not this fixed sleep

# Concurrency for rationale generation (see processSplitParallel in
# Preprocessing/rationaleGeneration.py). Kept LOW on purpose: Groq's free
# tier is the actual bottleneck (rate limiting starts around row ~300), and
# throwing more concurrent workers at a rate-limited free tier just produces
# more 429s in parallel, not more throughput. Raise this back to ~40 once
# the provider is a paid tier (DeepInfra) with real concurrent capacity.
RATIONALE_MAX_WORKERS = 20

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
# Common dimensionality that BERT's rationale-fused text vector, the Swin
# spatial stages, and the VGG19 frequency stages are all projected into
# before the Mlp-Mixer fuses them. Kept equal to EMB_DIM (768) by default so
# the text branch doesn't need an extra projection layer at all, but it's
# independent -- lower it (e.g. 256) to shrink the mixer and speed up
# training if you're compute constrained.
PFN_FUSION_DIM = EMB_DIM
 
# The paper uses batch size 12 and lr 0.001; the batch is heavier here than
# in the paper because every sample carries a BERT pass over 3 texts *and*
# a Swin *and* a VGG19 forward pass, so this defaults smaller/gentler than
# PFN's own paper settings -- raise PFN_BATCH_SIZE if your GPU has headroom.
PFN_BATCH_SIZE = 8
PFN_LR = 1e-4