# LORA Fine-Tuning for Sparse Attention

This module implements LORA (Low-Rank Adaptation) fine-tuning to teach the LLM to generate with sparse attention on NLI and QA tasks.

## Overview

The system compares model performance across 4 configurations:
1. **Base Model + CARG** (Regular attention)
2. **Base Model + SDAG** (Sparse attention without fine-tuning)
3. **Fine-tuned Model + CARG** (LORA fine-tuned on sparse attention data, evaluated with regular attention)
4. **Fine-tuned Model + SDAG** (LORA fine-tuned on sparse attention data, evaluated with sparse attention)

## Directory Structure

```
src/nli_sparse_attention/
├── train_and_eval.py          # Main CLI entry point
├── training_config.py          # Organized hyperparameters
├── data_preparation.py         # Dataset creation and loading
├── lora_trainer.py            # Training logic with LORA
├── evaluation.py              # Comprehensive evaluation
├── visualization.py           # Plotting utilities
├── nli_experiment.py          # Original experiment code
└── output/                    # All outputs
    ├── training_data/         # Saved datasets
    │   ├── training_data_nli.jsonl
    │   ├── validation_data_nli.jsonl
    │   ├── training_data_qa.jsonl
    │   ├── validation_data_qa.jsonl
    │   └── data_statistics.json
    ├── training/              # Training outputs
    │   ├── checkpoints/       # Model checkpoints
    │   │   ├── best_model/
    │   │   └── checkpoint-epoch-N/
    │   ├── plots/            # Training plots
    │   │   ├── loss_curve.png
    │   │   ├── learning_rate.png
    │   │   ├── perplexity.png
    │   │   └── training_summary.png
    │   └── logs/             # Training logs
    │       └── training_metrics.json
    └── evaluation/           # Evaluation outputs
        ├── results_*.csv
        ├── metrics_comparison_*.json
        └── plots/
            ├── accuracy_comparison_*.png
            ├── confusion_matrices_*.png
            └── f1_comparison_*.png
```

## Installation

Install required dependencies:

```bash
pip install torch transformers datasets peft bitsandbytes accelerate
pip install matplotlib seaborn tqdm
```

## Usage

### 1. Prepare Training Data

Create and save training/validation datasets:

```bash
# For NLI
python src/nli_sparse_attention/train_and_eval.py \
    --mode prepare-data \
    --experiment-mode nli

# For QA
python src/nli_sparse_attention/train_and_eval.py \
    --mode prepare-data \
    --experiment-mode qa
```

This will:
- Load data from SNLI (for NLI) or HotpotQA (for QA)
- Create training samples with sparse attention masks
- Save to `output/training_data/`
- Generate data statistics and visualizations

### 2. Train Model

Fine-tune the model with LORA:

```bash
# For NLI
python src/nli_sparse_attention/train_and_eval.py \
    --mode train \
    --experiment-mode nli

# For QA
python src/nli_sparse_attention/train_and_eval.py \
    --mode train \
    --experiment-mode qa
```

This will:
- Load training data from disk
- Setup LORA adapters on the base model
- Train with sparse attention masks
- Save checkpoints and best model
- Generate training plots (loss, learning rate, perplexity)

### 3. Evaluate Model

Evaluate base model and fine-tuned model:

```bash
# Evaluate with best model
python src/nli_sparse_attention/train_and_eval.py \
    --mode evaluate \
    --experiment-mode nli \
    --load-checkpoint best_model

# Evaluate with specific checkpoint
python src/nli_sparse_attention/train_and_eval.py \
    --mode evaluate \
    --experiment-mode qa \
    --load-checkpoint checkpoint-epoch-3
```

This will:
- Evaluate all 4 configurations on test set
- Generate detailed results CSV for each configuration
- Create comparison plots (accuracy, confusion matrices, F1 scores)
- Print comprehensive summary

### 4. Full Pipeline

Run everything in sequence:

```bash
python src/nli_sparse_attention/train_and_eval.py \
    --mode full-pipeline \
    --experiment-mode nli
```

This executes: data preparation → training → evaluation

### 5. Train and Evaluate Only

Skip data preparation if data already exists:

```bash
python src/nli_sparse_attention/train_and_eval.py \
    --mode train-and-evaluate \
    --experiment-mode nli
```

## Configuration

All hyperparameters are in [`training_config.py`](training_config.py), organized into sections:

### General Configuration
- `MODEL_NAME`: Base LLM model (default: `Qwen/Qwen2.5-7B-Instruct`)
- `DEVICE`: Computation device (default: `cuda:1`)
- `RANDOM_SEED`: For reproducibility (default: `42`)

### LORA Configuration
- `LORA_R`: Rank of low-rank matrices (default: `16`)
- `LORA_ALPHA`: Scaling factor (default: `32`)
- `LORA_DROPOUT`: Dropout probability (default: `0.1`)
- `LORA_TARGET_MODULES`: Layers to apply LORA (default: `["q_proj", "k_proj", "v_proj", "o_proj"]`)

### Training Configuration
- `NUM_EPOCHS`: Number of training epochs (default: `3`)
- `TRAIN_BATCH_SIZE`: Batch size (default: `4`)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation (default: `4`)
- `LEARNING_RATE`: Initial learning rate (default: `5e-5`)
- `WEIGHT_DECAY`: Weight decay (default: `0.01`)

### Optimizer Configuration (AdamW)
- `ADAM_BETA1`: Beta1 parameter (default: `0.9`)
- `ADAM_BETA2`: Beta2 parameter (default: `0.999`)
- `ADAM_EPSILON`: Epsilon parameter (default: `1e-8`)

### Learning Rate Scheduler
- `LR_SCHEDULER_TYPE`: Scheduler type (default: `"linear"`)
- `WARMUP_RATIO`: Warmup proportion (default: `0.1`)

### Data Configuration
- `TRAIN_SAMPLES_NLI`: Training samples for NLI (default: `6000`)
- `VAL_SAMPLES_NLI`: Validation samples for NLI (default: `1500`)
- `TEST_SAMPLES_NLI`: Test samples for NLI (default: `2000`)
- `TRAIN_SAMPLES_QA`: Training samples for QA (default: `6000`)
- `VAL_SAMPLES_QA`: Validation samples for QA (default: `1500`)
- `TEST_SAMPLES_QA`: Test samples for QA (default: `2000`)

### Evaluation Configuration
- `TEMPERATURE`: Sampling temperature (default: `0.1`)
- `TOP_P`: Nucleus sampling parameter (default: `1.0`)
- `MAX_NEW_TOKENS`: Maximum generation length (default: `250`)
- `EVAL_BATCH_SIZE`: Evaluation batch size (default: `8`)

## Command-Line Options

```
--mode                  Execution mode (required)
                        Choices: prepare-data, train, evaluate, train-and-evaluate, full-pipeline

--experiment-mode       Experiment type (required)
                        Choices: nli, qa

--model-name            Model identifier (default: from config)
--device                Device for computation (default: from config)
--load-checkpoint       Checkpoint to load for evaluation (e.g., 'best_model')
--force                 Force overwrite existing data/checkpoints
--seed                  Random seed (default: 42)
```

## Examples

### Example 1: Quick Start (NLI)

```bash
# Full pipeline for NLI
python src/nli_sparse_attention/train_and_eval.py \
    --mode full-pipeline \
    --experiment-mode nli
```

### Example 2: QA with Custom Model

```bash
# Prepare data
python src/nli_sparse_attention/train_and_eval.py \
    --mode prepare-data \
    --experiment-mode qa \
    --model-name meta-llama/Llama-3.1-8B-Instruct

# Train
python src/nli_sparse_attention/train_and_eval.py \
    --mode train \
    --experiment-mode qa \
    --model-name meta-llama/Llama-3.1-8B-Instruct

# Evaluate
python src/nli_sparse_attention/train_and_eval.py \
    --mode evaluate \
    --experiment-mode qa \
    --model-name meta-llama/Llama-3.1-8B-Instruct \
    --load-checkpoint best_model
```

### Example 3: Evaluate Only Base Model

```bash
# Evaluate base model without fine-tuning
python src/nli_sparse_attention/train_and_eval.py \
    --mode evaluate \
    --experiment-mode nli
# (Don't provide --load-checkpoint)
```

### Example 4: Regenerate Data

```bash
# Force regenerate training data
python src/nli_sparse_attention/train_and_eval.py \
    --mode prepare-data \
    --experiment-mode nli \
    --force
```

## Output Files

### Training Data
- `training_data_nli.jsonl` / `training_data_qa.jsonl`: Training samples with prompts, completions, and mask metadata
- `validation_data_nli.jsonl` / `validation_data_qa.jsonl`: Validation samples
- `data_statistics.json`: Dataset statistics (label distribution, length stats, etc.)
- `train_label_dist.png` / `val_label_dist.png`: Label distribution plots
- `length_statistics.png`: Prompt/answer length statistics

### Training Outputs
- `checkpoints/best_model/`: Best model checkpoint (lowest validation loss)
- `checkpoints/checkpoint-epoch-N/`: Periodic checkpoints
- `plots/loss_curve.png`: Training and validation loss over epochs
- `plots/learning_rate.png`: Learning rate schedule
- `plots/perplexity.png`: Perplexity over epochs
- `plots/training_summary.png`: 2x2 summary of all training metrics
- `logs/training_metrics.json`: Complete training history

### Evaluation Outputs
- `results_{config}_{timestamp}.csv`: Detailed predictions for each configuration
- `metrics_comparison_{timestamp}.json`: Comprehensive metrics comparison
- `plots/accuracy_comparison_{timestamp}.png`: Bar chart of accuracy/EM across configurations
- `plots/confusion_matrix_{config}_{timestamp}.png`: Confusion matrices (NLI only)
- `plots/f1_comparison_{timestamp}.png`: F1 score comparison (NLI only)

## Key Features

### 1. Data Persistence
- Training data is saved to disk after creation
- Reuse saved data across training runs
- Statistics and visualizations are auto-generated

### 2. Sparse Attention Training
- SDAG masks are applied during training
- Hypothesis cannot attend to premise (NLI)
- Documents are isolated from each other (QA)

### 3. LORA Efficiency
- 4-bit quantization with `bitsandbytes`
- Only trains small adapter matrices
- Preserves base model weights

### 4. Comprehensive Tracking
- Training loss, validation loss, perplexity
- Learning rate schedule, gradient norms
- All metrics saved to JSON

### 5. Automatic Visualization
- Training curves with train/val comparison
- Evaluation comparisons across 4 configurations
- Confusion matrices and F1 scores (NLI)

### 6. Early Stopping
- Monitors validation loss
- Saves best model automatically
- Configurable patience

### 7. Checkpoint Management
- Periodic checkpoints during training
- Best model saved separately
- Easy to resume or evaluate specific checkpoints

## Troubleshooting

### Out of Memory
- Reduce `TRAIN_BATCH_SIZE` in `training_config.py`
- Increase `GRADIENT_ACCUMULATION_STEPS`
- Enable 4-bit quantization: `USE_4BIT_QUANTIZATION = True`

### Slow Training
- Increase `TRAIN_BATCH_SIZE` if memory allows
- Reduce `TRAIN_SAMPLES_NLI` / `TRAIN_SAMPLES_QA` for faster iteration
- Use a smaller model (e.g., `Qwen/Qwen2.5-3B-Instruct`)

### Data Already Exists
- Use `--force` to regenerate data
- Or manually delete files in `output/training_data/`

### Checkpoint Not Found
- Check `output/training/checkpoints/` for available checkpoints
- Use `best_model` or `checkpoint-epoch-N` as checkpoint name

## References

- PEFT Library: https://github.com/huggingface/peft
- LORA Paper: https://arxiv.org/abs/2106.09685
- BitsAndBytes: https://github.com/TimDettmers/bitsandbytes
