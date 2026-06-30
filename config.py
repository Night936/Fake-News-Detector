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

#LLM Config based on Groq free tier rate limits
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
RATIONALE_MAX_RETRIES = 3
RATIONALE_SLEEP_BETWEEN_CALLS = 0.05  #seconds

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
 
USE_CUDA = True