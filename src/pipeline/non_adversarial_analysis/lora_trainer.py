"""
LORA Fine-Tuning Trainer with Sparse Attention

This module implements LORA fine-tuning with:
- Custom sparse attention masks during training
- AdamW optimizer with learning rate scheduling
- Comprehensive metrics tracking and visualization
- Checkpoint saving and loading
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BitsAndBytesConfig, get_linear_schedule_with_warmup

# Import from nli_experiment
from .nli_experiment import (
    build_nli_prompt,
    build_qa_prompt_and_spans,
    build_sdag_nli_mask,
    build_sdag_qa_doc_mask,
)

from . import training_config as training_config_module

# Import configuration
from .training_config import (
    # LORA config
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    LORA_BIAS,
    LORA_TASK_TYPE,
    # Quantization config
    USE_4BIT_QUANTIZATION,
    BNB_4BIT_COMPUTE_DTYPE,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    # Optimizer config
    LEARNING_RATE,
    WEIGHT_DECAY,
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPSILON,
    # Training config
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    # Scheduler config
    LR_SCHEDULER_TYPE,
    WARMUP_RATIO,
    # Evaluation config
    EVAL_EVERY_N_EPOCHS,
    SAVE_EVERY_N_EPOCHS,
    LOGGING_STEPS,
    SAVE_TOTAL_LIMIT,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_THRESHOLD,
    # Paths
    TRAINING_RUNS_DIR,
    TRAIN_RUN_NAME,
    # Other
    RANDOM_SEED,
    MODEL_DTYPE,
    DEVICE,
    get_total_training_steps,
    get_warmup_steps,
)

# Import visualization
from .visualization import (
    plot_training_metrics,
    create_training_summary_plot,
)

# Import data loading
from .data_preparation import load_data_from_jsonl


class SparseAttentionDataset(Dataset):
    """
    Dataset for training with sparse attention masks.

    Loads pre-processed data and rebuilds attention masks on-the-fly.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        mode: str,
        device: str = "cpu",
    ):
        """
        Initialize dataset.

        Args:
            data: List of pre-processed samples
            tokenizer: Tokenizer for the model
            mode: 'nli' or 'qa'
            device: Device for mask computation
        """
        self.data = data
        self.tokenizer = tokenizer
        self.mode = mode
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training sample.

        Returns:
            Dict with input_ids, attention_mask, labels, sparse_mask
        """
        sample = self.data[idx]

        # Tokenize prompt
        prompt = sample["prompt"]
        completion = sample["completion"]
        full_text = prompt + completion

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"][0]
        seq_len = input_ids.size(0)

        # Create labels (mask prompt tokens with -100)
        prompt_encoding = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_encoding["input_ids"].size(1)

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Don't compute loss on prompt tokens

        # Build sparse attention mask
        mask_metadata = sample["mask_metadata"]

        if self.mode == "nli":
            sparse_mask = build_sdag_nli_mask(
                seq_len=seq_len,
                system_user_len=mask_metadata["sys_user_len"],
                premise_start=mask_metadata["premise_start"],
                premise_end=mask_metadata["premise_end"],
                hypothesis_start=mask_metadata["hypothesis_start"],
                hypothesis_end=mask_metadata["hypothesis_end"],
                device=self.device,
            )
        elif self.mode == "qa":
            sparse_mask = build_sdag_qa_doc_mask(
                seq_len=seq_len,
                sys_user_len=mask_metadata["sys_user_len"],
                doc_token_spans=mask_metadata["doc_token_spans"],
                qa_start=mask_metadata["qa_start"],
                device=self.device,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "sparse_mask": sparse_mask,
        }


def collate_fn_sparse(batch: List[Dict]) -> Dict:
    """
    Custom collate function for sparse attention masks.

    Pads sequences to the same length within the batch.
    """
    # Find max length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)

    # Pad sequences
    input_ids_list = []
    labels_list = []
    sparse_masks_list = []
    attention_mask_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Pad input_ids
        input_ids = torch.cat([
            item["input_ids"],
            torch.zeros(pad_len, dtype=torch.long),
        ])
        input_ids_list.append(input_ids)

        # Pad labels
        labels = torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ])
        labels_list.append(labels)

        # Pad sparse mask (2D)
        sparse_mask = item["sparse_mask"]
        padded_mask = torch.zeros(max_len, max_len, dtype=torch.bool, device=sparse_mask.device)
        padded_mask[:seq_len, :seq_len] = sparse_mask
        sparse_masks_list.append(padded_mask)

        # Create attention mask (1D, for padding)
        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ])
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "sparse_masks": torch.stack(sparse_masks_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


class SparseAttentionTrainer:
    """
    Trainer for LORA fine-tuning with sparse attention masks.
    """

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset: SparseAttentionDataset,
        val_dataset: SparseAttentionDataset,
        mode: str,
        device: str = DEVICE,
        run_name: str = TRAIN_RUN_NAME,
    ):
        """
        Initialize trainer.

        Args:
            model: Base language model
            tokenizer: Tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset
            mode: 'nli' or 'qa'
            device: Device for training
            run_name: Custom run name prefix for output folder
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.mode = mode
        self.device = device
        self.run_name = self._sanitize_run_name(run_name)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_run_id = f"{self.run_name}_{self.run_timestamp}"
        self.run_id = base_run_id
        self.run_dir = os.path.join(TRAINING_RUNS_DIR, self.run_id)
        suffix = 1
        while os.path.exists(self.run_dir):
            self.run_id = f"{base_run_id}_{suffix:02d}"
            self.run_dir = os.path.join(TRAINING_RUNS_DIR, self.run_id)
            suffix += 1
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.logs_dir = os.path.join(self.run_dir, "logs")
        for directory in [self.run_dir, self.checkpoints_dir, self.plots_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Metrics history
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "train_perplexity": [],
            "val_perplexity": [],
            "epoch_numbers": [],
            "epoch_end_steps": [],
            "val_epochs": [],
            "val_step_indices": [],
            "train_loss_steps": [],
            "train_loss_step_indices": [],
            "learning_rate": [],
            "learning_rate_steps": [],
            "grad_norm": [],
            "grad_norm_steps": [],
            "val_loss_steps": [],
            "val_perplexity_steps": [],
        }

        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()

        # Create dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn_sparse,
        )

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=VAL_BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_sparse,
        )

        self.save_training_config_snapshot()

        print(f"Trainer initialized with {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        print(f"Run output directory: {self.run_dir}")

    @staticmethod
    def _sanitize_run_name(run_name: str) -> str:
        """Keep run names filesystem-safe and readable."""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name.strip())
        return safe or "run"

    @staticmethod
    def _serialize_config_value(value):
        """Convert config values to JSON-serializable form."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [SparseAttentionTrainer._serialize_config_value(v) for v in value]
        if isinstance(value, dict):
            return {
                str(k): SparseAttentionTrainer._serialize_config_value(v)
                for k, v in value.items()
            }
        return str(value)

    def save_training_config_snapshot(self):
        """Persist run-specific training configuration under logs directory."""
        config_values = {}
        for key, value in vars(training_config_module).items():
            if not key.isupper() or callable(value):
                continue
            config_values[key] = self._serialize_config_value(value)

        config_snapshot = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "run_timestamp": self.run_timestamp,
            "mode": self.mode,
            "device": self.device,
            "training_config": config_values,
        }

        config_json_path = os.path.join(self.logs_dir, "training_config.json")
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_snapshot, f, indent=2, sort_keys=True)

        config_source_path = getattr(training_config_module, "__file__", None)
        if config_source_path and os.path.isfile(config_source_path):
            config_py_path = os.path.join(self.logs_dir, "training_config.py")
            shutil.copy2(config_source_path, config_py_path)

        print(f"Training configuration saved to {config_json_path}")

    def setup_optimizer_and_scheduler(self):
        """Setup AdamW optimizer and learning rate scheduler."""
        # Get trainable parameters (only LORA parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        # Create optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=LEARNING_RATE,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPSILON,
            weight_decay=WEIGHT_DECAY,
        )

        # Create scheduler
        total_steps = get_total_training_steps(self.mode)
        warmup_steps = get_warmup_steps(self.mode)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
        print(f"Scheduler: Linear with warmup (warmup_steps={warmup_steps}, total_steps={total_steps})")

    def compute_loss_with_sparse_mask(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        sparse_masks: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss with sparse attention masks.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]
            sparse_masks: Sparse attention masks [batch_size, seq_len, seq_len]
            attention_mask: Padding mask [batch_size, seq_len]

        Returns:
            Loss tensor
        """
        batch_size, seq_len = input_ids.shape

        # Convert boolean masks to attention masks
        model_dtype = next(self.model.parameters()).dtype
        NEG_INF = torch.finfo(model_dtype).min

        # Convert sparse masks to attention masks
        attn_masks = torch.zeros_like(sparse_masks, dtype=model_dtype, device=self.device)
        attn_masks = attn_masks.masked_fill(~sparse_masks, NEG_INF)

        # Add batch and head dimensions: [batch_size, 1, seq_len, seq_len]
        attn_masks = attn_masks.unsqueeze(1)

        # Forward pass with custom attention masks
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attn_masks,
            labels=labels,
        )

        return outputs.loss

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, average_perplexity)
        """
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        num_batches = 0
        num_batches_total = len(self.train_dataloader)
        remainder_batches = num_batches_total % GRADIENT_ACCUMULATION_STEPS
        last_group_start = (num_batches_total - remainder_batches + 1) if remainder_batches else None
        accumulation_counter = 0
        accumulation_loss_total = 0.0

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch + 1}/{NUM_EPOCHS}")

        for step, batch in enumerate(progress_bar, start=1):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            sparse_masks = batch["sparse_masks"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Compute loss
            loss = self.compute_loss_with_sparse_mask(
                input_ids, labels, sparse_masks, attention_mask
            )

            # Scale loss for gradient accumulation (supports smaller final accumulation group)
            accumulation_divisor = GRADIENT_ACCUMULATION_STEPS
            if last_group_start is not None and step >= last_group_start:
                accumulation_divisor = remainder_batches

            step_loss = loss.item()
            scaled_loss = loss / accumulation_divisor
            scaled_loss.backward()

            total_loss += step_loss
            num_batches += 1
            accumulation_counter += 1
            accumulation_loss_total += step_loss

            # Update weights
            should_step = (step % GRADIENT_ACCUMULATION_STEPS == 0) or (step == num_batches_total)
            if should_step:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    MAX_GRAD_NORM
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                current_lr = self.scheduler.get_last_lr()[0]

                # Log metrics every optimizer step
                self.metrics_history["train_loss_steps"].append(step_loss)
                self.metrics_history["train_loss_step_indices"].append(self.global_step)
                self.metrics_history["learning_rate"].append(current_lr)
                self.metrics_history["learning_rate_steps"].append(self.global_step)
                self.metrics_history["grad_norm"].append(grad_norm.item())
                self.metrics_history["grad_norm_steps"].append(self.global_step)

                step_loss_for_log = accumulation_loss_total / max(accumulation_counter, 1)
                accumulation_counter = 0
                accumulation_loss_total = 0.0

                # Update progress bar
                if self.global_step % LOGGING_STEPS == 0:
                    progress_bar.set_postfix({
                        "loss": f"{step_loss_for_log:.4f}",
                        "lr": f"{current_lr:.2e}",
                    })

        avg_loss = total_loss / num_batches
        avg_perplexity = math.exp(min(avg_loss, 20))  # Clip to avoid overflow

        return avg_loss, avg_perplexity

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate on validation set.

        Returns:
            Tuple of (average_loss, average_perplexity)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            sparse_masks = batch["sparse_masks"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Compute loss
            loss = self.compute_loss_with_sparse_mask(
                input_ids, labels, sparse_masks, attention_mask
            )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_perplexity = math.exp(min(avg_loss, 20))

        return avg_loss, avg_perplexity

    def save_checkpoint(self, checkpoint_name: str = "checkpoint"):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save LORA adapter
        self.model.save_pretrained(checkpoint_path)

        # Save training state
        state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
            "metrics_history": self.metrics_history,
        }

        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        print(f"Checkpoint saved to {checkpoint_path}")

    def train(self) -> Dict:
        """
        Run full training loop.

        Returns:
            Dict with final metrics
        """
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Mode: {self.mode}")
        print(f"Run name: {self.run_name}")
        print(f"Run timestamp: {self.run_timestamp}")
        print(f"Run directory: {self.run_dir}")
        print(f"Epochs: {NUM_EPOCHS}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Train batch size: {TRAIN_BATCH_SIZE}")
        print(f"Validation batch size: {VAL_BATCH_SIZE}")
        print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
        print(f"Evaluation frequency: every {EVAL_EVERY_N_EPOCHS} epoch(s)")
        print(f"Checkpoint frequency: every {SAVE_EVERY_N_EPOCHS} epoch(s)")
        print("=" * 80 + "\n")

        for epoch in range(NUM_EPOCHS):
            self.current_epoch = epoch

            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"{'=' * 80}")

            # Train
            train_loss, train_ppl = self.train_epoch()
            self.metrics_history["train_loss"].append(train_loss)
            self.metrics_history["train_perplexity"].append(train_ppl)
            self.metrics_history["epoch_numbers"].append(epoch + 1)
            self.metrics_history["epoch_end_steps"].append(self.global_step)

            print(f"\nTraining - Loss: {train_loss:.4f}, Perplexity: {train_ppl:.2f}")

            should_evaluate = ((epoch + 1) % EVAL_EVERY_N_EPOCHS == 0) or (epoch + 1 == NUM_EPOCHS)
            if should_evaluate:
                val_loss, val_ppl = self.evaluate()
                self.metrics_history["val_loss"].append(val_loss)
                self.metrics_history["val_perplexity"].append(val_ppl)
                self.metrics_history["val_epochs"].append(epoch + 1)
                self.metrics_history["val_step_indices"].append(self.global_step)
                self.metrics_history["val_loss_steps"].append(val_loss)
                self.metrics_history["val_perplexity_steps"].append(val_ppl)

                print(f"Validation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

                # Check for improvement (best model tracking kept for future use)
                if val_loss < self.best_val_loss - EARLY_STOPPING_THRESHOLD:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    # self.save_checkpoint("best_model")  # Disabled: using last model instead
                    print("✓ New best validation loss!")
                else:
                    self.patience_counter += 1
                    print(f"No improvement ({self.patience_counter}/{EARLY_STOPPING_PATIENCE})")

            # Early stopping
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total epochs: {self.current_epoch + 1}")
        print("=" * 80 + "\n")

        # Save last model checkpoint
        self.save_checkpoint("last_model")
        print("✓ Last model saved!")

        # Save final metrics
        self.save_metrics()

        # Generate plots
        self.plot_metrics()

        return {
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.metrics_history["train_loss"][-1],
            "final_val_loss": self.metrics_history["val_loss"][-1] if self.metrics_history["val_loss"] else None,
            "total_epochs": self.current_epoch + 1,
            "run_name": self.run_name,
            "run_timestamp": self.run_timestamp,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "checkpoints_dir": self.checkpoints_dir,
            "last_model_path": os.path.join(self.checkpoints_dir, "last_model"),
        }

    def save_metrics(self):
        """Save training metrics to JSON."""
        metrics_file = os.path.join(self.logs_dir, "training_metrics.json")

        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        print(f"Training metrics saved to {metrics_file}")

    def plot_metrics(self):
        """Generate and save training plots."""
        print("Generating training plots...")

        # Individual plots
        plot_training_metrics(self.metrics_history, save_dir=self.plots_dir)

        # Summary plot
        summary_path = os.path.join(self.plots_dir, "training_summary.png")
        create_training_summary_plot(self.metrics_history, save_path=summary_path)

        print(f"Training plots saved to {self.plots_dir}")


def setup_model_for_training(base_model, tokenizer):
    """
    Setup model for LORA training with quantization.

    Args:
        base_model: Base language model
        tokenizer: Tokenizer

    Returns:
        PEFT model ready for training
    """
    print("Setting up model for LORA training...")

    # Prepare model for k-bit training
    if USE_4BIT_QUANTIZATION:
        model = prepare_model_for_kbit_training(base_model)
        print("✓ Model prepared for 4-bit training")
    else:
        model = base_model

    # Configure LORA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LORA
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def train_sparse_attention_lora(
    base_model,
    tokenizer,
    mode: str,
    device: str = DEVICE,
    run_name: str = TRAIN_RUN_NAME,
) -> Dict:
    """
    Main training function.

    Args:
        base_model: Base language model
        tokenizer: Tokenizer
        mode: 'nli' or 'qa'
        device: Device for training
        run_name: Custom run name prefix for output folder

    Returns:
        Dict with training results
    """
    # Load training data
    from .training_config import get_training_data_files
    train_file, val_file = get_training_data_files(mode)

    print(f"Loading training data from {train_file}...")
    train_data = load_data_from_jsonl(train_file)

    print(f"Loading validation data from {val_file}...")
    val_data = load_data_from_jsonl(val_file)

    # Create datasets
    train_dataset = SparseAttentionDataset(train_data, tokenizer, mode, device)
    val_dataset = SparseAttentionDataset(val_data, tokenizer, mode, device)

    # Setup model
    model = setup_model_for_training(base_model, tokenizer)
    model.to(device)

    # Create trainer
    trainer = SparseAttentionTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        mode=mode,
        device=device,
        run_name=run_name,
    )

    # Train
    results = trainer.train()

    return results
