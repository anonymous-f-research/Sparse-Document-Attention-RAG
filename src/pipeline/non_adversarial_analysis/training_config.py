"""
Training Configuration for LORA Fine-tuning with Sparse Attention

This module contains all hyperparameters organized into logical sections:
- General: model, device, paths
- NLI-specific: prompts, labels, parsing
- QA-specific: HotpotQA configuration
- Training: LORA, optimizer, scheduler
- Evaluation: generation parameters
"""

from __future__ import annotations

import os
import torch
from typing import Optional

# =========================
# GENERAL CONFIGURATION
# =========================

# Model Configuration
# Qwen/Qwen2.5-7B-Instruct meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Base LLM model
TRAIN_RUN_NAME = "Qwen"  # Custom name for run folder (combined with timestamp)
FALLBACK_MODEL_NAME: Optional[str] = None  # Fallback if MODEL_NAME unsupported
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"  # Device for inference
MODEL_DTYPE = torch.bfloat16 if "cuda" in DEVICE else torch.float32  # Model precision

# Base-Model Final Evaluation Controls
SKIP_BASE_EVAL = True  # If True, skip running base model task evaluation in final eval
BASE_EVAL_CSV_PATH: Optional[str] = "/lv_local/home/sagie.dekel/attention_effect_on_RAG/src/non_adversarial_setting/output/training/runs/qwen_20260408_140008/evaluation/"  # Existing base results CSV (or directory) to load metrics from

# Random Seed
RANDOM_SEED = 42  # For reproducibility

# Output Paths
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
TRAINING_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, "training_data")
TRAINING_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "training")
TRAINING_RUNS_DIR = os.path.join(TRAINING_OUTPUT_DIR, "runs")
EVALUATION_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "evaluation")
CHECKPOINTS_DIR = os.path.join(TRAINING_OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(TRAINING_OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(TRAINING_OUTPUT_DIR, "logs")

# Create directories
for directory in [BASE_OUTPUT_DIR, TRAINING_DATA_DIR, TRAINING_OUTPUT_DIR,
                  TRAINING_RUNS_DIR, EVALUATION_OUTPUT_DIR,
                  CHECKPOINTS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# =========================
# NLI-SPECIFIC CONFIGURATION
# =========================

# NLI Labels
NLI_LABELS = ["entailment", "neutral", "contradiction"]

# NLI Prompts
NLI_SYSTEM_PROMPT = """You are an NLP expert performing natural language inference on premise-hypothesis pairs.
Given a premise and a hypothesis, classify the hypothesis as:
- entailment
- neutral
- contradiction
with respect to the premise.
"""

NLI_USER_PROMPT_TEMPLATE_REASONING = """Premise: {premise}

Hypothesis: {hypothesis}

Determine the inference relation between the premise and the hypothesis.

Return your response as valid JSON with exactly these keys and order:
{{
  "explanation": "<concise reasoning in 1-2 sentences>",
  "answer": "<exactly one of: entailment, contradiction, neutral>"
}}
"""

NLI_USER_PROMPT_TEMPLATE_DIRECT = """Premise: {premise}
Hypothesis: {hypothesis}
Answer with exactly one of the options: entailment, contradiction, or neutral
Answer: """

# NLI Parsing Configuration
USE_REASONING_PROMPT = True  # True: JSON reasoning response, False: direct answer
USE_LLM_JUDGE = True  # Use LLM as judge for ambiguous answers

# Dataset Parameters for NLI
NLI_TOTAL_SAMPLES = 3000  # Total samples for evaluation
NLI_SAMPLES_PER_LABEL = NLI_TOTAL_SAMPLES // 3  # Equal distribution

# =========================
# QA-SPECIFIC CONFIGURATION
# =========================

# HotpotQA Configuration
HOTPOTQA_CONFIG = "fullwiki"
HOTPOTQA_SPLIT = "train"
HOTPOTQA_QUESTION_TYPE = "bridge"
HOTPOTQA_MIN_SUPPORTING_DOCS = 2  # Require at least this many supporting docs/facts

# QA Prompts
QA_SYSTEM_PROMPT = "You are a helpful assistant, below is a query from a user and some relevant contexts."
QA_USER_PROMPT_TEMPLATE = """Answer the question concisely, based on the following passages.

passages:
{docs_text}

- Question: {query}

- Answer: """

# =========================
# TRAINING CONFIGURATION
# =========================

# LORA Parameters
LORA_R = 32  # Rank of the low-rank matrices
LORA_ALPHA = 32  # Scaling factor for LORA weights
LORA_DROPOUT = 0.1  # Dropout probability for LORA layers
LORA_TARGET_MODULES = [
    "q_proj",  # Query projection
    "k_proj",  # Key projection
    "v_proj",  # Value projection
    "o_proj",  # Output projection
]
LORA_BIAS = "none"  # Bias handling: "none", "all", or "lora_only"
LORA_TASK_TYPE = "CAUSAL_LM"  # Task type for PEFT

# Quantization (for memory efficiency)
# Note: USE_4BIT_QUANTIZATION and USE_8BIT_QUANTIZATION are mutually exclusive.
USE_4BIT_QUANTIZATION = False   # Use 4-bit quantization (bitsandbytes)
USE_8BIT_QUANTIZATION = False  # Use 8-bit quantization; set True + USE_4BIT=False to enable
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"  # Compute dtype for 4-bit
BNB_4BIT_QUANT_TYPE = "nf4"  # Quantization type: "nf4" or "fp4"
BNB_4BIT_USE_DOUBLE_QUANT = True  # Use double quantization

# Training Data
TRAIN_SAMPLES_NLI = 6000  # Training samples for NLI
VAL_SAMPLES_NLI = 1500  # Validation samples for NLI
TRAIN_SAMPLES_QA = 1000  # Training samples for QA
VAL_SAMPLES_QA = 1000  # Validation samples for QA

# Data Files
TRAINING_DATA_NLI_FILE = os.path.join(TRAINING_DATA_DIR, "training_data_nli.jsonl")
VALIDATION_DATA_NLI_FILE = os.path.join(TRAINING_DATA_DIR, "validation_data_nli.jsonl")
TRAINING_DATA_QA_FILE = os.path.join(TRAINING_DATA_DIR, "training_data_qa.jsonl")
VALIDATION_DATA_QA_FILE = os.path.join(TRAINING_DATA_DIR, "validation_data_qa.jsonl")
DATA_STATISTICS_FILE = os.path.join(TRAINING_DATA_DIR, "data_statistics.json")

# Optimizer Parameters (AdamW)
LEARNING_RATE = 5e-4  # Initial learning rate
WEIGHT_DECAY = 0.01  # Weight decay for regularization
ADAM_BETA1 = 0.9  # Adam beta1
ADAM_BETA2 = 0.999  # Adam beta2
ADAM_EPSILON = 1e-8  # Adam epsilon

# Training Hyperparameters
NUM_EPOCHS = 3  # Number of training epochs
TRAIN_BATCH_SIZE = 4  # Batch size for training
VAL_BATCH_SIZE = 16  # Batch size for validation during training
GRADIENT_ACCUMULATION_STEPS = 6  # Gradient accumulation steps
EFFECTIVE_BATCH_SIZE = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS 
MAX_GRAD_NORM = 5.0  # Maximum gradient norm for clipping

# Learning Rate Scheduler
LR_SCHEDULER_TYPE = "cosine"  # "linear", "cosine", "constant", "constant_with_warmup"
WARMUP_RATIO = 0.1  # Proportion of training steps for warmup 

# Evaluation During Training (epoch-based)
EVAL_EVERY_N_EPOCHS = 1  # Evaluate every N epochs
SAVE_EVERY_N_EPOCHS = 1  # Save periodic checkpoint every N epochs
LOGGING_STEPS = 10  # Log optimization metrics every N optimizer steps
SAVE_TOTAL_LIMIT = 1  # Keep only last N checkpoints

# Early Stopping
EARLY_STOPPING_PATIENCE = 1000  # Stop if no improvement for N evaluations
EARLY_STOPPING_THRESHOLD = 0.001  # Minimum improvement threshold

# Mixed Precision Training
FP16 = False  # Use FP16 mixed precision (not compatible with bfloat16)
BF16 = True if MODEL_DTYPE == torch.bfloat16 else False  # Use BF16 mixed precision

# =========================
# EVALUATION CONFIGURATION (for training and final eval)
# =========================

# Generation Parameters
TEMPERATURE = 0.1  # Temperature for sampling (lower = more deterministic)
TOP_P = 1.0  # Nucleus sampling parameter
MAX_NEW_TOKENS = 250  # Maximum tokens to generate per sample

# Evaluation Batch Size
EVAL_BATCH_SIZE = 100  # Batch size for evaluation (can be larger than training)

# Test Set Sizes (for final evaluation)
TEST_SAMPLES_NLI = 2000  # Test samples for NLI
TEST_SAMPLES_QA = 1000  # Test samples for QA

# Batch Processing
#BATCH_SIZE = 64  # Number of samples to process in parallel (for non-training eval)

# WikiText Perplexity Evaluation
ENABLE_WIKITEXT_PERPLEXITY_EVAL = True  # Evaluate LM perplexity on WikiText during final evaluation
WIKITEXT_DATASET_NAME = "Salesforce/wikitext"  # HuggingFace dataset name
WIKITEXT_CONFIG_NAME = "wikitext-2-raw-v1"  # Dataset configuration
WIKITEXT_SPLIT = "test"  # Split used for perplexity evaluation
WIKITEXT_NUM_SAMPLES = 5000  # Number of WikiText samples to evaluate
WIKITEXT_MAX_LENGTH = 256  # Max tokens per sample for perplexity evaluation
WIKITEXT_SDAG_SPLIT_RATIO = 0.5  # Split point for SDAG mask on plain text



# Evaluation Debug Printing
EVAL_DEBUG_PRINT_SAMPLES = True  # Print sample-level debug info during final evaluation
EVAL_DEBUG_NUM_SAMPLES_PER_CONFIG = 1  # Number of samples to print per evaluated configuration
EVAL_DEBUG_PRINT_PROMPT = True  # Print the full rendered prompt (truncated by max chars)
EVAL_DEBUG_MAX_TEXT_CHARS = 1500  # Max chars per long text field in debug printing

# =========================
# DEBUG CONFIGURATION
# =========================

# Debug Configuration (prompt split + attention mask inspection)
DEBUG_PRINT_SPLITS_AND_MASK = True
DEBUG_PRINT_MAX_SAMPLES = 1
DEBUG_MASK_MAX_TOKENS = 120  # Print at most this many rows/cols from the mask
DEBUG_PRINT_EXAMPLE_IO = True
DEBUG_PRINT_EXAMPLE_MAX = 1

# =========================
# EXPERIMENT MODE
# =========================

EXPERIMENT_MODE = "qa"  # "nli" or "qa"

# =========================
# HELPER FUNCTIONS
# =========================

def get_nli_user_prompt_template(use_reasoning: bool = USE_REASONING_PROMPT) -> str:
    """Get NLI user prompt template based on reasoning mode."""
    return NLI_USER_PROMPT_TEMPLATE_REASONING if use_reasoning else NLI_USER_PROMPT_TEMPLATE_DIRECT


def get_training_data_files(mode: str) -> tuple[str, str]:
    """Get training and validation data file paths for given mode."""
    if mode == "nli":
        return TRAINING_DATA_NLI_FILE, VALIDATION_DATA_NLI_FILE
    elif mode == "qa":
        return TRAINING_DATA_QA_FILE, VALIDATION_DATA_QA_FILE
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'nli' or 'qa'.")


def get_num_training_samples(mode: str) -> tuple[int, int]:
    """Get number of training and validation samples for given mode."""
    if mode == "nli":
        return TRAIN_SAMPLES_NLI, VAL_SAMPLES_NLI
    elif mode == "qa":
        return TRAIN_SAMPLES_QA, VAL_SAMPLES_QA
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'nli' or 'qa'.")


def get_total_training_steps(mode: str) -> int:
    """Calculate total training steps."""
    train_samples, _ = get_num_training_samples(mode)
    steps_per_epoch = train_samples // (TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    return steps_per_epoch * NUM_EPOCHS


def get_warmup_steps(mode: str) -> int:
    """Calculate number of warmup steps."""
    total_steps = get_total_training_steps(mode)
    return int(total_steps * WARMUP_RATIO)
