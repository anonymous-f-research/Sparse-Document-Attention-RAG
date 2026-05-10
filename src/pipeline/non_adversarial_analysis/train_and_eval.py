#!/usr/bin/env python3
"""
Main Script for LORA Fine-Tuning and Evaluation

This script provides multiple modes:
- prepare-data: Create and save training data
- train: Fine-tune model with LORA
- evaluate: Evaluate model(s) on test set
- train-and-evaluate: Train then evaluate
- full-pipeline: Prepare data, train, and evaluate

Usage:
    python train_and_eval.py --mode prepare-data --experiment-mode nli
    python train_and_eval.py --mode train --experiment-mode nli --use-saved-data
    python train_and_eval.py --mode evaluate --experiment-mode qa --load-checkpoint best_model
    python train_and_eval.py --mode train-and-evaluate --experiment-mode qa --skip-base-eval --base-eval-csv /path/to/results_base_carg_*.csv
    python train_and_eval.py --mode full-pipeline --experiment-mode nli
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from .training_config import (
    MODEL_NAME,
    FALLBACK_MODEL_NAME,
    DEVICE,
    MODEL_DTYPE,
    RANDOM_SEED,
    CHECKPOINTS_DIR,
    TRAINING_OUTPUT_DIR,
    TRAINING_RUNS_DIR,
    USE_4BIT_QUANTIZATION,
    USE_8BIT_QUANTIZATION,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    SKIP_BASE_EVAL,
    BASE_EVAL_CSV_PATH,
)

from .data_preparation import (
    prepare_and_save_training_data,
    check_saved_data_exists,
    load_statistics,
)

from .lora_trainer import train_sparse_attention_lora

from .evaluation import run_comprehensive_evaluation

from .visualization import plot_data_statistics


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str):
    """Load tokenizer with proper configuration."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except AttributeError as e:
        if "has no attribute 'keys'" in str(e):
            print("Tokenizer compatibility fallback: retrying with extra_special_tokens={}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, extra_special_tokens={})
        else:
            raise

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(model_name: str, use_quantization: bool = USE_4BIT_QUANTIZATION):
    """Load base model with optional 4-bit or 8-bit quantization."""
    print(f"Loading model: {model_name}...")

    if use_quantization and USE_8BIT_QUANTIZATION:
        # Setup 8-bit quantization
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=DEVICE,
            trust_remote_code=True,
        )
        print("✓ Model loaded with 8-bit quantization")
    elif use_quantization and USE_4BIT_QUANTIZATION:
        # Setup 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=getattr(torch, BNB_4BIT_COMPUTE_DTYPE),
            bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=DEVICE,
            trust_remote_code=True,
        )
        print("✓ Model loaded with 4-bit quantization")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=MODEL_DTYPE,
            device_map=DEVICE,
            trust_remote_code=True,
        )
        print("✓ Model loaded")

    model.eval()

    # Ensure generation config has padding token
    if getattr(model.generation_config, "pad_token_id", None) is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    return model


def resolve_checkpoint_path(checkpoint_ref: str) -> Optional[str]:
    """Resolve checkpoint reference to an existing path."""
    if not checkpoint_ref:
        return None

    candidates = []

    if os.path.isabs(checkpoint_ref):
        candidates.append(checkpoint_ref)
    else:
        # User-provided relative path
        candidates.append(checkpoint_ref)
        # Legacy global checkpoint dir
        candidates.append(os.path.join(CHECKPOINTS_DIR, checkpoint_ref))
        # Relative to training output root
        candidates.append(os.path.join(TRAINING_OUTPUT_DIR, checkpoint_ref))
        # Relative to runs root (<run_id>/checkpoints/best_model etc.)
        candidates.append(os.path.join(TRAINING_RUNS_DIR, checkpoint_ref))

        # Latest-run fallback for simple names like "best_model"
        if os.path.isdir(TRAINING_RUNS_DIR):
            run_dirs = [entry.path for entry in os.scandir(TRAINING_RUNS_DIR) if entry.is_dir()]
            run_dirs.sort(key=os.path.getmtime, reverse=True)
            for run_dir in run_dirs:
                candidates.append(os.path.join(run_dir, "checkpoints", checkpoint_ref))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None


def infer_run_evaluation_dir(checkpoint_path: Optional[str]) -> Optional[str]:
    """Infer run-scoped evaluation output directory from a checkpoint path."""
    if not checkpoint_path:
        return None

    abs_checkpoint_path = os.path.abspath(checkpoint_path)
    abs_runs_dir = os.path.abspath(TRAINING_RUNS_DIR)

    try:
        rel_path = os.path.relpath(abs_checkpoint_path, abs_runs_dir)
    except ValueError:
        return None

    if rel_path.startswith(os.pardir):
        return None

    run_id = rel_path.split(os.sep, 1)[0]
    run_dir = os.path.join(abs_runs_dir, run_id)
    if not os.path.isdir(run_dir):
        return None

    return os.path.join(run_dir, "evaluation")


def mode_prepare_data(args):
    """Mode: Prepare training data."""
    print("\n" + "=" * 80)
    print("MODE: PREPARE DATA")
    print("=" * 80)

    # Check if data already exists
    if check_saved_data_exists(args.experiment_mode) and not args.force:
        print(f"\nTraining data for {args.experiment_mode} already exists!")
        print("Use --force to overwrite existing data.")
        return

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name)

    # Prepare and save data
    train_file, val_file = prepare_and_save_training_data(
        tokenizer=tokenizer,
        mode=args.experiment_mode,
        device=args.device,
    )

    # Load and visualize statistics
    try:
        from .training_config import DATA_STATISTICS_FILE, TRAINING_DATA_DIR
        stats = load_statistics(DATA_STATISTICS_FILE)

        print("\nGenerating data visualization...")
        plot_data_statistics(stats, save_dir=TRAINING_DATA_DIR)
        print(f"Data statistics plots saved to {TRAINING_DATA_DIR}")

    except Exception as e:
        print(f"Warning: Could not generate data statistics plots: {e}")

    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)


def mode_train(args) -> Dict:
    """Mode: Train model with LORA."""
    print("\n" + "=" * 80)
    print("MODE: TRAIN")
    print("=" * 80)

    # Check if training data exists
    if not check_saved_data_exists(args.experiment_mode):
        print(f"\nError: Training data for {args.experiment_mode} not found!")
        print("Please run with --mode prepare-data first.")
        sys.exit(1)

    # Load model and tokenizer
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(args.model_name, use_quantization=USE_4BIT_QUANTIZATION)

    # Train
    results = train_sparse_attention_lora(
        base_model=base_model,
        tokenizer=tokenizer,
        mode=args.experiment_mode,
        device=args.device,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final training loss: {results['final_train_loss']:.4f}")
    print(f"Total epochs: {results['total_epochs']}")
    print(f"Run directory: {results['run_dir']}")
    print(f"Model saved to: {results['last_model_path']}")
    print("=" * 80)
    return results


def mode_evaluate(args, evaluation_output_dir: Optional[str] = None) -> Dict:
    """Mode: Evaluate model(s)."""
    print("\n" + "=" * 80)
    print("MODE: EVALUATE")
    print("=" * 80)

    # Load model and tokenizer
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_base_model(args.model_name, use_quantization=False)  # No quantization for eval

    # Determine fine-tuned model path
    finetuned_path = None
    if args.load_checkpoint:
        checkpoint_path = resolve_checkpoint_path(args.load_checkpoint)
        if checkpoint_path:
            finetuned_path = checkpoint_path
            print(f"Will evaluate fine-tuned model from: {finetuned_path}")
        else:
            print(f"Warning: Checkpoint '{args.load_checkpoint}' not found. Will only evaluate base model.")

    target_eval_output_dir = evaluation_output_dir or infer_run_evaluation_dir(finetuned_path)
    if target_eval_output_dir:
        print(f"Evaluation artifacts will be saved to: {target_eval_output_dir}")

    if args.base_eval_csv and not args.skip_base_eval:
        print("Note: --base-eval-csv was provided but --skip-base-eval is not enabled; CSV backfill will be ignored.")

    # Run evaluation
    results = run_comprehensive_evaluation(
        base_model=base_model,
        tokenizer=tokenizer,
        mode=args.experiment_mode,
        device=args.device,
        finetuned_model_path=finetuned_path,
        output_dir=target_eval_output_dir,
        skip_base_eval=args.skip_base_eval,
        base_eval_csv_path=args.base_eval_csv,
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved under: {results['output_dir']}")
    return results


def mode_train_and_evaluate(args):
    """Mode: Train then evaluate."""
    print("\n" + "=" * 80)
    print("MODE: TRAIN AND EVALUATE")
    print("=" * 80)

    # Train
    train_results = mode_train(args)

    # Evaluate with the last model
    args.load_checkpoint = train_results["last_model_path"]
    run_eval_dir = os.path.join(train_results["run_dir"], "evaluation")
    mode_evaluate(args, evaluation_output_dir=run_eval_dir)


def mode_full_pipeline(args):
    """Mode: Prepare data, train, and evaluate."""
    print("\n" + "=" * 80)
    print("MODE: FULL PIPELINE")
    print("=" * 80)

    # Prepare data
    if not check_saved_data_exists(args.experiment_mode) or args.force:
        mode_prepare_data(args)
    else:
        print(f"\nTraining data for {args.experiment_mode} already exists. Skipping data preparation.")
        print("Use --force to regenerate data.")

    # Train
    train_results = mode_train(args)

    # Evaluate
    args.load_checkpoint = train_results["last_model_path"]
    run_eval_dir = os.path.join(train_results["run_dir"], "evaluation")
    mode_evaluate(args, evaluation_output_dir=run_eval_dir)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LORA Fine-Tuning and Evaluation for Sparse Attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["prepare-data", "train", "evaluate", "train-and-evaluate", "full-pipeline"],
        help="Execution mode",
    )

    parser.add_argument(
        "--experiment-mode",
        type=str,
        required=True,
        choices=["nli", "qa"],
        help="Experiment mode: nli or qa",
    )

    # Optional arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help=f"Model name (default: {MODEL_NAME})",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device for computation (default: {DEVICE})",
    )

    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Checkpoint name to load for evaluation (e.g., 'best_model', 'checkpoint-epoch-3')",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing data/checkpoints",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})",
    )

    parser.add_argument(
        "--skip-base-eval",
        action=argparse.BooleanOptionalAction,
        default=SKIP_BASE_EVAL,
        help=f"Skip base-model task evaluation in final evaluation stage (default: {SKIP_BASE_EVAL}).",
    )

    parser.add_argument(
        "--base-eval-csv",
        type=str,
        default=BASE_EVAL_CSV_PATH,
        help=(
            "Optional path to existing base evaluation CSV (or evaluation directory) "
            "to backfill base metrics when --skip-base-eval is used."
        ),
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Route to appropriate mode
    if args.mode == "prepare-data":
        mode_prepare_data(args)
    elif args.mode == "train":
        mode_train(args)
    elif args.mode == "evaluate":
        mode_evaluate(args)
    elif args.mode == "train-and-evaluate":
        mode_train_and_evaluate(args)
    elif args.mode == "full-pipeline":
        mode_full_pipeline(args)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
