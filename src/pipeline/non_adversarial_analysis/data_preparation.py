"""
Data Preparation Module for LORA Fine-tuning

This module handles:
- Creating training datasets from SNLI and HotpotQA
- Saving datasets to disk (JSONL format)
- Loading saved datasets
- Computing and saving dataset statistics
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

# Import from nli_experiment
from .nli_experiment import (
    build_nli_prompt,
    build_qa_prompt_and_spans,
    build_sdag_nli_mask,
    build_sdag_qa_doc_mask,
    extract_supporting_fact_docs,
    NLI_LABELS,
)

# Import configuration
from .training_config import (
    RANDOM_SEED,
    TRAIN_SAMPLES_NLI,
    VAL_SAMPLES_NLI,
    TRAIN_SAMPLES_QA,
    VAL_SAMPLES_QA,
    TRAINING_DATA_NLI_FILE,
    VALIDATION_DATA_NLI_FILE,
    TRAINING_DATA_QA_FILE,
    VALIDATION_DATA_QA_FILE,
    DATA_STATISTICS_FILE,
    HOTPOTQA_CONFIG,
    HOTPOTQA_SPLIT,
    HOTPOTQA_QUESTION_TYPE,
    HOTPOTQA_MIN_SUPPORTING_DOCS,
    get_nli_user_prompt_template,
    NLI_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT_TEMPLATE,
)


def load_snli_for_training(
    total_samples: int,
    seed: int = RANDOM_SEED,
    split: str = "train",
) -> List[Dict[str, str]]:
    """
    Load SNLI dataset for training with balanced labels.

    Args:
        total_samples: Total number of samples to load
        seed: Random seed for reproducibility
        split: Dataset split ('train', 'validation', 'test')

    Returns:
        List of dicts with keys: premise, hypothesis, label
    """
    print(f"Loading SNLI {split} dataset from HuggingFace...")
    dataset = load_dataset("stanfordnlp/snli", split=split)

    # Label mapping: 0=entailment, 1=neutral, 2=contradiction
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    # Group by label
    by_label = defaultdict(list)
    for item in dataset:
        if item["label"] in [0, 1, 2]:  # Filter out -1 (no gold label)
            by_label[item["label"]].append(item)

    # Sample from each label
    np.random.seed(seed)
    samples_per_label = total_samples // 3
    balanced_samples = []

    for label_id in [0, 1, 2]:
        available = by_label[label_id]
        if len(available) < samples_per_label:
            print(f"Warning: Only {len(available)} samples available for {label_map[label_id]}, requested {samples_per_label}")
            sampled = available
        else:
            indices = np.random.choice(len(available), samples_per_label, replace=False)
            sampled = [available[i] for i in indices]

        for item in sampled:
            balanced_samples.append({
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": label_map[label_id],
            })

    # Shuffle
    np.random.shuffle(balanced_samples)

    print(f"Loaded {len(balanced_samples)} balanced samples from SNLI {split}")
    label_counts = {label: sum(1 for s in balanced_samples if s["label"] == label) for label in NLI_LABELS}
    print(f"Label distribution: {label_counts}")

    return balanced_samples


def load_hotpotqa_for_training(
    total_samples: int,
    seed: int = RANDOM_SEED,
    config_name: str = HOTPOTQA_CONFIG,
    split_name: str = HOTPOTQA_SPLIT,
    question_type: str = HOTPOTQA_QUESTION_TYPE,
    min_supporting_docs: int = HOTPOTQA_MIN_SUPPORTING_DOCS,
) -> List[Dict[str, str]]:
    """
    Load HotpotQA dataset for training.

    Args:
        total_samples: Total number of samples to load
        seed: Random seed
        config_name: HotpotQA config
        split_name: Dataset split
        question_type: Question type filter
        min_supporting_docs: Minimum number of supporting docs required

    Returns:
        List of dicts with keys: id, question, answer, supporting_docs
    """
    print(
        f"Loading HotpotQA ({config_name}/{split_name}) from HuggingFace "
        f"with min_supporting_docs={min_supporting_docs}..."
    )
    dataset = load_dataset("hotpot_qa", config_name, split=split_name)

    samples: List[Dict[str, str]] = []
    for item in dataset:
        if item.get("type") != question_type:
            continue
        docs = extract_supporting_fact_docs(item)
        if len(docs) < min_supporting_docs:
            continue
        samples.append({
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "supporting_docs": docs,
        })

    if not samples:
        raise ValueError(
            f"No HotpotQA samples with at least {min_supporting_docs} supporting docs were found."
        )

    np.random.seed(seed)
    if len(samples) > total_samples:
        indices = np.random.choice(len(samples), total_samples, replace=False)
        sampled = [samples[i] for i in indices]
    else:
        sampled = samples
        print(f"Warning: only {len(sampled)} samples available (requested {total_samples})")

    np.random.shuffle(sampled)
    print(f"Loaded {len(sampled)} HotpotQA {question_type} samples")
    return sampled


def create_nli_training_sample(
    sample: Dict[str, str],
    tokenizer,
    device: str = "cpu",
) -> Dict:
    """
    Create a training sample for NLI with SDAG sparse attention.

    Args:
        sample: Raw SNLI sample with premise, hypothesis, label
        tokenizer: Tokenizer for the model
        device: Device for mask computation

    Returns:
        Dict with prompt, completion, mask_metadata
    """
    premise = sample["premise"]
    hypothesis = sample["hypothesis"]
    label = sample["label"]

    # Build prompt and get token spans
    chat_str, sys_user_len, premise_start, premise_end, hypothesis_start, hypothesis_end = build_nli_prompt(
        tokenizer=tokenizer,
        premise=premise,
        hypothesis=hypothesis,
        system_prompt=NLI_SYSTEM_PROMPT,
        user_template=get_nli_user_prompt_template(),
    )

    # The completion is just the label (for direct mode) or JSON response (for reasoning mode)
    from .training_config import USE_REASONING_PROMPT
    if USE_REASONING_PROMPT:
        completion = json.dumps({
            "explanation": f"The hypothesis is {label} given the premise.",
            "answer": label
        })
    else:
        completion = label

    # Store mask metadata (we'll rebuild masks during training)
    mask_metadata = {
        "sys_user_len": sys_user_len,
        "premise_start": premise_start,
        "premise_end": premise_end,
        "hypothesis_start": hypothesis_start,
        "hypothesis_end": hypothesis_end,
    }

    return {
        "prompt": chat_str,
        "completion": completion,
        "mask_metadata": mask_metadata,
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
    }


def create_qa_training_sample(
    sample: Dict[str, str],
    tokenizer,
    device: str = "cpu",
) -> Dict:
    """
    Create a training sample for QA with SDAG sparse attention.

    Args:
        sample: Raw HotpotQA sample with question, answer, supporting_docs
        tokenizer: Tokenizer for the model
        device: Device for mask computation

    Returns:
        Dict with prompt, completion, mask_metadata
    """
    question = sample["question"]
    answer = sample["answer"]
    supporting_docs = sample["supporting_docs"]

    # Build prompt and get token spans
    chat_str, sys_user_len, doc_token_spans, qa_start, doc_lines = build_qa_prompt_and_spans(
        tokenizer=tokenizer,
        question=question,
        supporting_docs=supporting_docs,
        system_prompt=QA_SYSTEM_PROMPT,
        user_template=QA_USER_PROMPT_TEMPLATE,
    )

    # Completion is the answer
    completion = answer

    # Store mask metadata
    mask_metadata = {
        "sys_user_len": sys_user_len,
        "doc_token_spans": doc_token_spans,
        "qa_start": qa_start,
    }

    return {
        "prompt": chat_str,
        "completion": completion,
        "mask_metadata": mask_metadata,
        "question": question,
        "answer": answer,
        "supporting_docs": supporting_docs,
        "id": sample.get("id", ""),
    }


def prepare_nli_training_data(
    tokenizer,
    train_samples: int = TRAIN_SAMPLES_NLI,
    val_samples: int = VAL_SAMPLES_NLI,
    device: str = "cpu",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare NLI training and validation data.

    Args:
        tokenizer: Tokenizer for the model
        train_samples: Number of training samples
        val_samples: Number of validation samples
        device: Device for computation

    Returns:
        Tuple of (train_data, val_data)
    """
    print("Preparing NLI training data...")

    # Load training data from SNLI train split
    train_raw = load_snli_for_training(total_samples=train_samples, split="train")

    # Load validation data from SNLI validation split
    val_raw = load_snli_for_training(total_samples=val_samples, split="validation")

    # Create training samples
    print("Creating training samples with SDAG masks...")
    train_data = []
    for sample in tqdm(train_raw, desc="Processing training samples"):
        train_data.append(create_nli_training_sample(sample, tokenizer, device))

    print("Creating validation samples with SDAG masks...")
    val_data = []
    for sample in tqdm(val_raw, desc="Processing validation samples"):
        val_data.append(create_nli_training_sample(sample, tokenizer, device))

    return train_data, val_data


def prepare_qa_training_data(
    tokenizer,
    train_samples: int = TRAIN_SAMPLES_QA,
    val_samples: int = VAL_SAMPLES_QA,
    device: str = "cpu",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Prepare QA training and validation data.

    Args:
        tokenizer: Tokenizer for the model
        train_samples: Number of training samples
        val_samples: Number of validation samples
        device: Device for computation

    Returns:
        Tuple of (train_data, val_data)
    """
    print("Preparing QA training data...")

    # Load all samples
    total_samples = train_samples + val_samples
    all_samples = load_hotpotqa_for_training(total_samples=total_samples)

    # Split into train/val
    train_raw = all_samples[:train_samples]
    val_raw = all_samples[train_samples:]

    # Create training samples
    print("Creating training samples with SDAG masks...")
    train_data = []
    for sample in tqdm(train_raw, desc="Processing training samples"):
        train_data.append(create_qa_training_sample(sample, tokenizer, device))

    print("Creating validation samples with SDAG masks...")
    val_data = []
    for sample in tqdm(val_raw, desc="Processing validation samples"):
        val_data.append(create_qa_training_sample(sample, tokenizer, device))

    return train_data, val_data


def save_data_to_jsonl(data: List[Dict], filepath: str) -> None:
    """Save data to JSONL file."""
    print(f"Saving {len(data)} samples to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved successfully!")


def load_data_from_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSONL file."""
    print(f"Loading data from {filepath}...")
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} samples")
    return data


def compute_data_statistics(
    train_data: List[Dict],
    val_data: List[Dict],
    mode: str,
) -> Dict:
    """
    Compute statistics for the dataset.

    Args:
        train_data: Training data
        val_data: Validation data
        mode: 'nli' or 'qa'

    Returns:
        Dict with statistics
    """
    stats = {
        "mode": mode,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "total_size": len(train_data) + len(val_data),
    }

    if mode == "nli":
        # Label distribution
        train_labels = [d["label"] for d in train_data]
        val_labels = [d["label"] for d in val_data]
        stats["train_label_distribution"] = dict(Counter(train_labels))
        stats["val_label_distribution"] = dict(Counter(val_labels))

        # Prompt length statistics
        train_prompt_lens = [len(d["prompt"]) for d in train_data]
        val_prompt_lens = [len(d["prompt"]) for d in val_data]
        stats["train_prompt_length"] = {
            "mean": float(np.mean(train_prompt_lens)),
            "std": float(np.std(train_prompt_lens)),
            "min": int(np.min(train_prompt_lens)),
            "max": int(np.max(train_prompt_lens)),
        }
        stats["val_prompt_length"] = {
            "mean": float(np.mean(val_prompt_lens)),
            "std": float(np.std(val_prompt_lens)),
            "min": int(np.min(val_prompt_lens)),
            "max": int(np.max(val_prompt_lens)),
        }

    elif mode == "qa":
        # Answer length statistics
        train_answer_lens = [len(d["answer"]) for d in train_data]
        val_answer_lens = [len(d["answer"]) for d in val_data]
        stats["train_answer_length"] = {
            "mean": float(np.mean(train_answer_lens)),
            "std": float(np.std(train_answer_lens)),
            "min": int(np.min(train_answer_lens)),
            "max": int(np.max(train_answer_lens)),
        }
        stats["val_answer_length"] = {
            "mean": float(np.mean(val_answer_lens)),
            "std": float(np.std(val_answer_lens)),
            "min": int(np.min(val_answer_lens)),
            "max": int(np.max(val_answer_lens)),
        }

        # Number of supporting docs
        train_num_docs = [len(d["supporting_docs"]) for d in train_data]
        val_num_docs = [len(d["supporting_docs"]) for d in val_data]
        stats["train_num_supporting_docs"] = {
            "mean": float(np.mean(train_num_docs)),
            "std": float(np.std(train_num_docs)),
            "min": int(np.min(train_num_docs)),
            "max": int(np.max(train_num_docs)),
        }
        stats["val_num_supporting_docs"] = {
            "mean": float(np.mean(val_num_docs)),
            "std": float(np.std(val_num_docs)),
            "min": int(np.min(val_num_docs)),
            "max": int(np.max(val_num_docs)),
        }

    return stats


def save_statistics(stats: Dict, filepath: str = DATA_STATISTICS_FILE) -> None:
    """Save statistics to JSON file."""
    print(f"Saving statistics to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("Statistics saved!")


def load_statistics(filepath: str = DATA_STATISTICS_FILE) -> Dict:
    """Load statistics from JSON file."""
    print(f"Loading statistics from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats


def prepare_and_save_training_data(
    tokenizer,
    mode: str,
    device: str = "cpu",
) -> Tuple[str, str]:
    """
    Prepare training data and save to disk.

    Args:
        tokenizer: Tokenizer for the model
        mode: 'nli' or 'qa'
        device: Device for computation

    Returns:
        Tuple of (train_file_path, val_file_path)
    """
    if mode == "nli":
        train_data, val_data = prepare_nli_training_data(tokenizer, device=device)
        train_file = TRAINING_DATA_NLI_FILE
        val_file = VALIDATION_DATA_NLI_FILE
    elif mode == "qa":
        train_data, val_data = prepare_qa_training_data(tokenizer, device=device)
        train_file = TRAINING_DATA_QA_FILE
        val_file = VALIDATION_DATA_QA_FILE
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'nli' or 'qa'.")

    # Save data
    save_data_to_jsonl(train_data, train_file)
    save_data_to_jsonl(val_data, val_file)

    # Compute and save statistics
    stats = compute_data_statistics(train_data, val_data, mode)
    save_statistics(stats)

    # Print summary
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Training file: {train_file}")
    print(f"Validation file: {val_file}")
    print(f"Statistics file: {DATA_STATISTICS_FILE}")
    print("=" * 80)

    return train_file, val_file


def check_saved_data_exists(mode: str) -> bool:
    """Check if saved training data exists for the given mode."""
    if mode == "nli":
        return os.path.exists(TRAINING_DATA_NLI_FILE) and os.path.exists(VALIDATION_DATA_NLI_FILE)
    elif mode == "qa":
        return os.path.exists(TRAINING_DATA_QA_FILE) and os.path.exists(VALIDATION_DATA_QA_FILE)
    else:
        raise ValueError(f"Invalid mode: {mode}. Expected 'nli' or 'qa'.")
