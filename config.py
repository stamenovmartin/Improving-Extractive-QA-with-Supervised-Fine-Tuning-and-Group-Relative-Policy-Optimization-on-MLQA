import torch
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = False  # NaN со flan-t5-small, мора FP32

GENERATOR_MODEL = "google/flan-t5-small"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 256
TOP_K = 3  # број на retrieved chunks за RAG

# SFT
SFT_EPOCHS = 3
SFT_LR = 5e-5
SFT_BATCH_SIZE = 8
SFT_WARMUP_RATIO = 0.1
SFT_MAX_GRAD_NORM = 1.0

# GRPO
GRPO_GROUP_SIZE = 4         # K кандидати по prompt
GRPO_LR = 1e-5
GRPO_EPOCHS = 3
GRPO_BATCH_SIZE = 4
GRPO_CLIP_EPS = 0.2
GRPO_KL_BETA = 0.04
GRPO_MAX_GRAD_NORM = 1.0
GRPO_WARMUP_RATIO = 0.1
GRPO_TEMPERATURE = 0.8
GRPO_TOP_P = 0.9

# Reward
REWARD_F1_WEIGHT = 0.7
REWARD_EM_WEIGHT = 0.3
REWARD_LENGTH_PENALTY = 0.001
REWARD_MAX_LENGTH = 64

MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 64

DATA_SEED = 42
TRAIN_RATIO = 0.9

EVAL_EVERY_STEPS = 200
EARLY_STOPPING_PATIENCE = 3
EVAL_TEST_SAMPLES = 500

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLQA_DIR = os.path.join(BASE_DIR, "mlqa_data", "MLQA_V1")
MLQA_DEV = os.path.join(MLQA_DIR, "dev", "dev-context-en-question-en.json")
MLQA_TEST = os.path.join(MLQA_DIR, "test", "test-context-en-question-en.json")

SFT_MODEL_DIR = os.path.join(BASE_DIR, "models", "sft")
GRPO_BEST_DIR = os.path.join(BASE_DIR, "models", "grpo_best")
GRPO_FINAL_DIR = os.path.join(BASE_DIR, "models", "grpo_final")

PLOTS_DIR = os.path.join(BASE_DIR, "plots")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
