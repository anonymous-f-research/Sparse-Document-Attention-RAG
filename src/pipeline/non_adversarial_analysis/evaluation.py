"""
Comprehensive Evaluation Module

Compares model performance across 4 configurations:
1. Base model + CARG (regular attention)
2. Base model + SDAG (sparse attention)
3. Fine-tuned model + CARG
4. Fine-tuned model + SDAG

Generates detailed metrics, plots, and comparison reports.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm

# Import from nli_experiment
from .nli_experiment import (
    build_nli_prompt,
    build_qa_prompt_and_spans,
    build_sdag_nli_mask,
    build_sdag_qa_doc_mask,
    generate_with_custom_mask,
    parse_nli_answer,
    qa_exact_match,
    load_balanced_snli,
    load_hotpotqa_bridge,
    NLI_LABELS,
)

# Import configuration
from .training_config import (
    EVALUATION_OUTPUT_DIR,
    TEST_SAMPLES_NLI,
    TEST_SAMPLES_QA,
    EVAL_BATCH_SIZE,
    TEMPERATURE,
    TOP_P,
    MAX_NEW_TOKENS,
    RANDOM_SEED,
    CHECKPOINTS_DIR,
    get_nli_user_prompt_template,
    NLI_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT_TEMPLATE,
    USE_REASONING_PROMPT,
    USE_LLM_JUDGE,
    EVAL_DEBUG_PRINT_SAMPLES,
    EVAL_DEBUG_NUM_SAMPLES_PER_CONFIG,
    EVAL_DEBUG_PRINT_PROMPT,
    EVAL_DEBUG_MAX_TEXT_CHARS,
    ENABLE_WIKITEXT_PERPLEXITY_EVAL,
    WIKITEXT_DATASET_NAME,
    WIKITEXT_CONFIG_NAME,
    WIKITEXT_SPLIT,
    WIKITEXT_NUM_SAMPLES,
    WIKITEXT_MAX_LENGTH,
    WIKITEXT_SDAG_SPLIT_RATIO,
)

# Import visualization
from .visualization import (
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_f1_scores_comparison,
    plot_perplexity_comparison,
)


def _truncate_for_debug(text: str, max_chars: int = EVAL_DEBUG_MAX_TEXT_CHARS) -> str:
    """Truncate long text values for concise debug output."""
    value = "" if text is None else str(text)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    clipped = len(value) - max_chars
    return f"{value[:max_chars]}... [truncated {clipped} chars]"


def _print_eval_sample_debug(
    *,
    mode: str,
    config_name: str,
    sample_index: int,
    use_sparse: bool,
    prompt: str,
    prediction: str,
    raw_answer: str,
    score_value: int,
    premise: Optional[str] = None,
    hypothesis: Optional[str] = None,
    true_label: Optional[str] = None,
    question: Optional[str] = None,
    true_answer: Optional[str] = None,
    supporting_docs: Optional[List[str]] = None,
) -> None:
    """Print one evaluation sample for debugging."""
    attention_name = "SDAG" if use_sparse else "CARG"
    print("\n" + "=" * 80)
    print(f"EVAL DEBUG SAMPLE | Config={config_name} | Sample={sample_index} | Attention={attention_name}")

    if mode == "nli":
        print(f"Premise: {_truncate_for_debug(premise)}")
        print(f"Hypothesis: {_truncate_for_debug(hypothesis)}")
        print(f"True label: {true_label}")
        print(f"Prediction: {prediction}")
        print(f"Correct: {score_value}")
    else:
        print(f"Question: {_truncate_for_debug(question)}")
        print(f"True answer: {_truncate_for_debug(true_answer)}")
        print(f"Prediction: {_truncate_for_debug(prediction)}")
        print(f"Exact match: {score_value}")
        if supporting_docs:
            print(f"Supporting docs count: {len(supporting_docs)}")
            print(f"Supporting doc [0]: {_truncate_for_debug(supporting_docs[0])}")
            if len(supporting_docs) > 1:
                print(f"Supporting doc [1]: {_truncate_for_debug(supporting_docs[1])}")

    if EVAL_DEBUG_PRINT_PROMPT:
        print("\nPrompt:")
        print(_truncate_for_debug(prompt))

    print("\nRaw answer:")
    print(_truncate_for_debug(raw_answer))
    print("=" * 80)


def load_wikitext_samples(
    num_samples: int = WIKITEXT_NUM_SAMPLES,
    seed: int = RANDOM_SEED,
) -> List[str]:
    """Load non-empty WikiText samples for perplexity evaluation."""
    print(
        f"Loading WikiText samples: dataset={WIKITEXT_DATASET_NAME}, "
        f"config={WIKITEXT_CONFIG_NAME}, split={WIKITEXT_SPLIT}, num_samples={num_samples}"
    )

    from datasets import load_dataset

    dataset = load_dataset(WIKITEXT_DATASET_NAME, WIKITEXT_CONFIG_NAME, split=WIKITEXT_SPLIT)
    texts = [str(item.get("text", "")).strip() for item in dataset]
    texts = [t for t in texts if t]

    if not texts:
        raise ValueError("No non-empty WikiText samples were found.")

    np.random.seed(seed)
    if len(texts) > num_samples:
        indices = np.random.choice(len(texts), size=num_samples, replace=False)
        sampled = [texts[i] for i in indices]
    else:
        sampled = texts
        print(f"Warning: only {len(sampled)} non-empty samples available (requested {num_samples})")

    np.random.shuffle(sampled)
    print(f"Loaded {len(sampled)} WikiText samples for perplexity evaluation")
    return sampled


def build_sdag_text_mask(
    seq_len: int,
    split_ratio: float = WIKITEXT_SDAG_SPLIT_RATIO,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build SDAG mask for plain text by isolating the second segment from the first segment.

    Behavior:
    - First segment: standard causal attention.
    - Second segment: standard causal inside second segment, but blocked from first-segment content.
      Token 0 remains visible to all tokens as a lightweight shared prefix anchor.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")

    split_idx = int(seq_len * split_ratio)
    split_idx = max(1, min(split_idx, seq_len - 1))

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # First segment: regular causal.
    for i in range(split_idx):
        mask[i, : i + 1] = True

    # Second segment: allow BOS/prefix token + in-segment causal; block first-segment content.
    for i in range(split_idx, seq_len):
        mask[i, 0] = True
        mask[i, split_idx : i + 1] = True

    return mask


@torch.no_grad()
def evaluate_wikitext_perplexity(
    model,
    tokenizer,
    wikitext_samples: List[str],
    device: str,
    use_sparse: bool,
) -> Dict[str, float]:
    """Compute token-level perplexity on WikiText for CARG or SDAG attention."""
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    evaluated_samples = 0

    model_dtype = next(model.parameters()).dtype
    neg_inf = torch.finfo(model_dtype).min

    desc = "WikiText PPL (SDAG)" if use_sparse else "WikiText PPL (CARG)"
    for text in tqdm(wikitext_samples, desc=desc):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=WIKITEXT_MAX_LENGTH,
        )
        input_ids = encoded["input_ids"].to(device)

        if input_ids.size(1) < 2:
            continue

        labels = input_ids.clone()

        if use_sparse:
            seq_len = input_ids.size(1)
            sparse_mask = build_sdag_text_mask(
                seq_len=seq_len,
                split_ratio=WIKITEXT_SDAG_SPLIT_RATIO,
                device=device,
            )
            attention_mask = torch.zeros_like(sparse_mask, dtype=model_dtype, device=device)
            attention_mask = attention_mask.masked_fill(~sparse_mask, neg_inf).unsqueeze(0).unsqueeze(1)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            attention_mask = torch.ones_like(input_ids, device=device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        # CausalLM loss is computed on shifted labels -> predicted tokens are positions [1:].
        token_count = input_ids.size(1) - 1
        total_nll += float(outputs.loss.item()) * token_count
        total_tokens += token_count
        evaluated_samples += 1

    if total_tokens == 0:
        raise ValueError("No valid WikiText samples were evaluated for perplexity.")

    avg_nll = total_nll / total_tokens
    perplexity = math.exp(min(avg_nll, 20))

    return {
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "total_tokens": total_tokens,
        "total_samples": evaluated_samples,
        "use_sparse": int(use_sparse),
    }


def load_test_data_nli(
    num_samples: int = TEST_SAMPLES_NLI,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """Load NLI test data from SNLI test split."""
    print(f"Loading {num_samples} NLI test samples...")

    # Use the test split to ensure no overlap with training
    from datasets import load_dataset
    from collections import defaultdict

    dataset = load_dataset("stanfordnlp/snli", split="test")
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    # Group by label
    by_label = defaultdict(list)
    for item in dataset:
        if item["label"] in [0, 1, 2]:
            by_label[item["label"]].append(item)

    # Sample from each label
    np.random.seed(seed)
    samples_per_label = num_samples // 3
    test_samples = []

    for label_id in [0, 1, 2]:
        available = by_label[label_id]
        if len(available) < samples_per_label:
            sampled = available
        else:
            indices = np.random.choice(len(available), samples_per_label, replace=False)
            sampled = [available[i] for i in indices]

        for item in sampled:
            test_samples.append({
                "premise": item["premise"],
                "hypothesis": item["hypothesis"],
                "label": label_map[label_id],
            })

    np.random.shuffle(test_samples)
    print(f"Loaded {len(test_samples)} test samples")

    return test_samples


def load_test_data_qa(
    num_samples: int = TEST_SAMPLES_QA,
    seed: int = RANDOM_SEED,
) -> List[Dict]:
    """Load QA test data from HotpotQA."""
    print(f"Loading {num_samples} QA test samples...")

    # Load from validation split to avoid overlap with training (which uses train split)
    from .training_config import HOTPOTQA_CONFIG, HOTPOTQA_MIN_SUPPORTING_DOCS
    samples = load_hotpotqa_bridge(
        total_samples=num_samples,
        seed=seed,
        config_name=HOTPOTQA_CONFIG,
        split_name="validation",  # Use validation split for testing
        min_supporting_docs=HOTPOTQA_MIN_SUPPORTING_DOCS,
    )

    return samples


def evaluate_nli_batch(
    model,
    tokenizer,
    batch_samples: List[Dict],
    device: str,
    use_sparse: bool = False,
    debug_state: Optional[Dict] = None,
    start_sample_idx: int = 0,
    config_name: str = "",
) -> List[Dict]:
    """
    Evaluate a batch of NLI samples.

    Args:
        model: Language model
        tokenizer: Tokenizer
        batch_samples: List of NLI samples
        device: Device for inference
        use_sparse: Whether to use SDAG sparse attention

    Returns:
        List of result dicts
    """
    results = []

    for local_idx, sample in enumerate(batch_samples):
        premise = sample["premise"]
        hypothesis = sample["hypothesis"]
        true_label = sample["label"]

        # Build prompt
        chat_str, sys_user_len, premise_start, premise_end, hypothesis_start, hypothesis_end = build_nli_prompt(
            tokenizer=tokenizer,
            premise=premise,
            hypothesis=hypothesis,
            system_prompt=NLI_SYSTEM_PROMPT,
            user_template=get_nli_user_prompt_template(),
        )

        # Tokenize
        encoded = tokenizer(chat_str, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        seq_len = input_ids.size(1)

        if use_sparse:
            # SDAG: sparse attention
            sdag_mask = build_sdag_nli_mask(
                seq_len=seq_len,
                system_user_len=sys_user_len,
                premise_start=premise_start,
                premise_end=premise_end,
                hypothesis_start=hypothesis_start,
                hypothesis_end=hypothesis_end,
                device=device,
            )
            answer = generate_with_custom_mask(
                model, tokenizer, input_ids, prompt_mask=sdag_mask,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
            )
        else:
            # CARG: regular attention
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=TEMPERATURE > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()

        # Parse answer
        prediction = parse_nli_answer(
            answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_llm_judge=USE_LLM_JUDGE,
            use_reasoning_prompt=USE_REASONING_PROMPT,
        )

        result = {
            "premise": premise,
            "hypothesis": hypothesis,
            "true_label": true_label,
            "prediction": prediction,
            "raw_answer": answer,
            "correct": int(prediction == true_label),
        }
        results.append(result)

        if debug_state and debug_state.get("enabled") and debug_state.get("printed", 0) < debug_state.get("max_samples", 0):
            _print_eval_sample_debug(
                mode="nli",
                config_name=config_name,
                sample_index=start_sample_idx + local_idx,
                use_sparse=use_sparse,
                prompt=chat_str,
                prediction=prediction,
                raw_answer=answer,
                score_value=result["correct"],
                premise=premise,
                hypothesis=hypothesis,
                true_label=true_label,
            )
            debug_state["printed"] = debug_state.get("printed", 0) + 1

    return results


def evaluate_qa_batch(
    model,
    tokenizer,
    batch_samples: List[Dict],
    device: str,
    use_sparse: bool = False,
    debug_state: Optional[Dict] = None,
    start_sample_idx: int = 0,
    config_name: str = "",
) -> List[Dict]:
    """
    Evaluate a batch of QA samples.

    Args:
        model: Language model
        tokenizer: Tokenizer
        batch_samples: List of QA samples
        device: Device for inference
        use_sparse: Whether to use SDAG sparse attention

    Returns:
        List of result dicts
    """
    results = []

    for local_idx, sample in enumerate(batch_samples):
        question = sample["question"]
        true_answer = sample["answer"]
        supporting_docs = sample["supporting_docs"]

        # Build prompt
        chat_str, sys_user_len, doc_token_spans, qa_start, doc_lines = build_qa_prompt_and_spans(
            tokenizer=tokenizer,
            question=question,
            supporting_docs=supporting_docs,
            system_prompt=QA_SYSTEM_PROMPT,
            user_template=QA_USER_PROMPT_TEMPLATE,
        )

        # Tokenize
        encoded = tokenizer(chat_str, return_tensors="pt").to(device)
        input_ids = encoded["input_ids"]
        seq_len = input_ids.size(1)

        if use_sparse:
            # SDAG: sparse attention
            sdag_mask = build_sdag_qa_doc_mask(
                seq_len=seq_len,
                sys_user_len=sys_user_len,
                doc_token_spans=doc_token_spans,
                qa_start=qa_start,
                device=device,
            )
            answer = generate_with_custom_mask(
                model, tokenizer, input_ids, prompt_mask=sdag_mask,
                max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE
            )
        else:
            # CARG: regular attention
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=TEMPERATURE > 0,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()

        # Compute exact match
        exact_match = int(qa_exact_match(answer, true_answer))

        result = {
            "id": sample.get("id", ""),
            "question": question,
            "true_answer": true_answer,
            "prediction": answer,
            "raw_answer": answer,
            "exact_match": exact_match,
        }
        results.append(result)

        if debug_state and debug_state.get("enabled") and debug_state.get("printed", 0) < debug_state.get("max_samples", 0):
            _print_eval_sample_debug(
                mode="qa",
                config_name=config_name,
                sample_index=start_sample_idx + local_idx,
                use_sparse=use_sparse,
                prompt=chat_str,
                prediction=answer,
                raw_answer=answer,
                score_value=exact_match,
                question=question,
                true_answer=true_answer,
                supporting_docs=supporting_docs,
            )
            debug_state["printed"] = debug_state.get("printed", 0) + 1

    return results


def compute_nli_metrics(results: List[Dict]) -> Dict:
    """Compute metrics for NLI results."""
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total > 0 else 0.0

    # Per-label metrics
    per_label = {}
    for label in NLI_LABELS:
        tp = sum(1 for r in results if r["true_label"] == label and r["prediction"] == label)
        fp = sum(1 for r in results if r["true_label"] != label and r["prediction"] == label)
        fn = sum(1 for r in results if r["true_label"] == label and r["prediction"] != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    # Confusion matrix
    confusion = {true_l: {pred_l: 0 for pred_l in NLI_LABELS + ["unknown"]} for true_l in NLI_LABELS}
    for r in results:
        true_l = r["true_label"]
        pred_l = r["prediction"]
        if pred_l not in confusion[true_l]:
            confusion[true_l]["unknown"] = confusion[true_l].get("unknown", 0) + 1
        else:
            confusion[true_l][pred_l] += 1

    return {
        "accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "per_label_metrics": per_label,
        "confusion_matrix": confusion,
    }


def compute_qa_metrics(results: List[Dict]) -> Dict:
    """Compute metrics for QA results."""
    total = len(results)
    exact_matches = sum(r["exact_match"] for r in results)
    em_score = exact_matches / total if total > 0 else 0.0

    return {
        "exact_match": em_score,
        "total_samples": total,
        "exact_match_count": exact_matches,
    }


def _infer_base_config_key_from_csv_path(csv_path: str) -> Optional[str]:
    """Infer base configuration key from a saved results CSV filename."""
    filename = os.path.basename(csv_path).lower()
    if "base_carg" in filename:
        return "base_carg"
    if "base_sdag" in filename:
        return "base_sdag"
    return None


def _load_results_from_saved_eval_csv(csv_path: str, mode: str) -> List[Dict]:
    """Load saved evaluation CSV rows and restore typed result records."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results: List[Dict] = []
    if mode == "nli":
        for row in rows:
            results.append(
                {
                    "premise": row.get("premise", ""),
                    "hypothesis": row.get("hypothesis", ""),
                    "true_label": row.get("true_label", ""),
                    "prediction": row.get("prediction", ""),
                    "raw_answer": row.get("raw_answer", ""),
                    "correct": int(row.get("correct", 0) or 0),
                }
            )
    else:
        for row in rows:
            results.append(
                {
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "true_answer": row.get("true_answer", ""),
                    "prediction": row.get("prediction", ""),
                    "raw_answer": row.get("raw_answer", ""),
                    "exact_match": int(row.get("exact_match", 0) or 0),
                }
            )

    return results


def load_base_metrics_from_existing_csv(
    mode: str,
    base_eval_csv_path: str,
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """
    Load base-model results/metrics from existing evaluation CSV file(s).

    Supports either:
    - A single CSV path named like results_base_carg_*.csv or results_base_sdag_*.csv
    - A directory containing those files (latest per configuration is used)
    """
    if not base_eval_csv_path:
        raise ValueError("base_eval_csv_path is required when loading metrics from CSV.")

    source_path = os.path.abspath(base_eval_csv_path)
    config_to_csv: Dict[str, str] = {}

    if os.path.isdir(source_path):
        for config_key in ("base_carg", "base_sdag"):
            pattern = os.path.join(source_path, f"results_{config_key}_*.csv")
            matches = glob.glob(pattern)
            if not matches:
                continue
            matches.sort(key=os.path.getmtime, reverse=True)
            config_to_csv[config_key] = matches[0]
    else:
        inferred = _infer_base_config_key_from_csv_path(source_path)
        if inferred is None:
            raise ValueError(
                f"Could not infer base config from CSV filename: {source_path}. "
                "Expected filename containing 'base_carg' or 'base_sdag'."
            )
        config_to_csv[inferred] = source_path

    if not config_to_csv:
        raise ValueError(
            f"No base evaluation CSV files found under path: {source_path}. "
            "Provide either a base CSV file path or an evaluation directory."
        )

    loaded_results: Dict[str, List[Dict]] = {}
    loaded_metrics: Dict[str, Dict] = {}

    for config_key, csv_path in config_to_csv.items():
        results = _load_results_from_saved_eval_csv(csv_path, mode=mode)
        if mode == "nli":
            metrics = compute_nli_metrics(results)
        else:
            metrics = compute_qa_metrics(results)

        metrics["loaded_from_csv"] = os.path.abspath(csv_path)
        loaded_results[config_key] = results
        loaded_metrics[config_key] = metrics

        metric_key = "accuracy" if mode == "nli" else "exact_match"
        print(
            f"Loaded base metrics from CSV: config={config_key}, "
            f"score={metrics[metric_key]:.4f}, path={csv_path}"
        )

    return loaded_results, loaded_metrics


def evaluate_model_configuration(
    model,
    tokenizer,
    test_data: List[Dict],
    mode: str,
    device: str,
    use_sparse: bool,
    config_name: str,
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate a single model configuration.

    Args:
        model: Language model
        tokenizer: Tokenizer
        test_data: Test dataset
        mode: 'nli' or 'qa'
        device: Device for inference
        use_sparse: Whether to use SDAG
        config_name: Name of this configuration

    Returns:
        Tuple of (results, metrics)
    """
    print(f"\nEvaluating: {config_name}")
    print(f"Mode: {mode}, Sparse: {use_sparse}")
    print(f"Test samples: {len(test_data)}")

    debug_state = {
        "enabled": EVAL_DEBUG_PRINT_SAMPLES,
        "max_samples": EVAL_DEBUG_NUM_SAMPLES_PER_CONFIG,
        "printed": 0,
    }
    if debug_state["enabled"]:
        print(
            f"Debug printing enabled: up to {debug_state['max_samples']} sample(s) for {config_name}"
        )

    all_results = []

    # Process in batches
    num_batches = (len(test_data) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE

    with tqdm(total=len(test_data), desc=config_name) as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * EVAL_BATCH_SIZE
            end_idx = min(start_idx + EVAL_BATCH_SIZE, len(test_data))
            batch = test_data[start_idx:end_idx]

            if mode == "nli":
                batch_results = evaluate_nli_batch(
                    model=model,
                    tokenizer=tokenizer,
                    batch_samples=batch,
                    device=device,
                    use_sparse=use_sparse,
                    debug_state=debug_state,
                    start_sample_idx=start_idx,
                    config_name=config_name,
                )
            else:
                batch_results = evaluate_qa_batch(
                    model=model,
                    tokenizer=tokenizer,
                    batch_samples=batch,
                    device=device,
                    use_sparse=use_sparse,
                    debug_state=debug_state,
                    start_sample_idx=start_idx,
                    config_name=config_name,
                )

            all_results.extend(batch_results)
            pbar.update(len(batch_results))

    # Compute metrics
    if mode == "nli":
        metrics = compute_nli_metrics(all_results)
        metric_value = metrics["accuracy"]
    else:
        metrics = compute_qa_metrics(all_results)
        metric_value = metrics["exact_match"]

    print(f"✓ {config_name}: {metric_value:.4f}")

    return all_results, metrics


def run_comprehensive_evaluation(
    base_model,
    tokenizer,
    mode: str,
    device: str,
    finetuned_model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    skip_base_eval: bool = False,
    base_eval_csv_path: Optional[str] = None,
) -> Dict:
    """
    Run comprehensive evaluation across all configurations.

    Args:
        base_model: Base language model
        tokenizer: Tokenizer
        mode: 'nli' or 'qa'
        device: Device for inference
        finetuned_model_path: Path to fine-tuned LORA adapter (optional)
        output_dir: Directory to save evaluation artifacts (optional)
        skip_base_eval: Skip running base-model task evaluation (CARG/SDAG)
        base_eval_csv_path: Optional CSV path (or directory) to load base metrics from

    Returns:
        Dict with all results and metrics
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 80)

    # Load test data
    if mode == "nli":
        test_data = load_test_data_nli()
    else:
        test_data = load_test_data_qa()

    wikitext_samples: Optional[List[str]] = None
    if ENABLE_WIKITEXT_PERPLEXITY_EVAL:
        wikitext_samples = load_wikitext_samples()

    all_results = {}
    all_metrics = {}
    target_output_dir = output_dir or EVALUATION_OUTPUT_DIR

    if skip_base_eval:
        print("\n" + "-" * 80)
        print("Skipping base-model task evaluation (--skip-base-eval enabled)")
        print("-" * 80)

        if base_eval_csv_path:
            loaded_results, loaded_metrics = load_base_metrics_from_existing_csv(
                mode=mode,
                base_eval_csv_path=base_eval_csv_path,
            )
            all_results.update(loaded_results)
            all_metrics.update(loaded_metrics)
        else:
            print("No base CSV provided; base configurations will be absent from this run's comparison.")
    else:
        # Configuration 1: Base model + CARG
        print("\n" + "-" * 80)
        print("Configuration 1: Base Model + CARG (Regular Attention)")
        print("-" * 80)
        results, metrics = evaluate_model_configuration(
            base_model, tokenizer, test_data, mode, device,
            use_sparse=False, config_name="Base + CARG"
        )
        if wikitext_samples is not None:
            ppl_info = evaluate_wikitext_perplexity(
                model=base_model,
                tokenizer=tokenizer,
                wikitext_samples=wikitext_samples,
                device=device,
                use_sparse=False,
            )
            metrics["wikitext_perplexity"] = ppl_info["perplexity"]
            metrics["wikitext_perplexity_details"] = ppl_info
            print(f"✓ Base + CARG WikiText PPL: {ppl_info['perplexity']:.6f}")
        all_results["base_carg"] = results
        all_metrics["base_carg"] = metrics

        # Configuration 2: Base model + SDAG
        print("\n" + "-" * 80)
        print("Configuration 2: Base Model + SDAG (Sparse Attention)")
        print("-" * 80)
        results, metrics = evaluate_model_configuration(
            base_model, tokenizer, test_data, mode, device,
            use_sparse=True, config_name="Base + SDAG"
        )
        if wikitext_samples is not None:
            ppl_info = evaluate_wikitext_perplexity(
                model=base_model,
                tokenizer=tokenizer,
                wikitext_samples=wikitext_samples,
                device=device,
                use_sparse=True,
            )
            metrics["wikitext_perplexity"] = ppl_info["perplexity"]
            metrics["wikitext_perplexity_details"] = ppl_info
            print(f"✓ Base + SDAG WikiText PPL: {ppl_info['perplexity']:.6f}")
        all_results["base_sdag"] = results
        all_metrics["base_sdag"] = metrics

    # Load fine-tuned model if available
    if finetuned_model_path:
        print(f"\nLoading fine-tuned model from {finetuned_model_path}...")
        finetuned_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
        finetuned_model.to(device)
        finetuned_model.eval()
        print("✓ Fine-tuned model loaded")

        # Configuration 3: Fine-tuned + CARG
        print("\n" + "-" * 80)
        print("Configuration 3: Fine-tuned Model + CARG (Regular Attention)")
        print("-" * 80)
        results, metrics = evaluate_model_configuration(
            finetuned_model, tokenizer, test_data, mode, device,
            use_sparse=False, config_name="Fine-tuned + CARG"
        )
        if wikitext_samples is not None:
            ppl_info = evaluate_wikitext_perplexity(
                model=finetuned_model,
                tokenizer=tokenizer,
                wikitext_samples=wikitext_samples,
                device=device,
                use_sparse=False,
            )
            metrics["wikitext_perplexity"] = ppl_info["perplexity"]
            metrics["wikitext_perplexity_details"] = ppl_info
            print(f"✓ Fine-tuned + CARG WikiText PPL: {ppl_info['perplexity']:.6f}")
        all_results["finetuned_carg"] = results
        all_metrics["finetuned_carg"] = metrics

        # Configuration 4: Fine-tuned + SDAG
        print("\n" + "-" * 80)
        print("Configuration 4: Fine-tuned Model + SDAG (Sparse Attention)")
        print("-" * 80)
        results, metrics = evaluate_model_configuration(
            finetuned_model, tokenizer, test_data, mode, device,
            use_sparse=True, config_name="Fine-tuned + SDAG"
        )
        if wikitext_samples is not None:
            ppl_info = evaluate_wikitext_perplexity(
                model=finetuned_model,
                tokenizer=tokenizer,
                wikitext_samples=wikitext_samples,
                device=device,
                use_sparse=True,
            )
            metrics["wikitext_perplexity"] = ppl_info["perplexity"]
            metrics["wikitext_perplexity_details"] = ppl_info
            print(f"✓ Fine-tuned + SDAG WikiText PPL: {ppl_info['perplexity']:.6f}")
        all_results["finetuned_sdag"] = results
        all_metrics["finetuned_sdag"] = metrics

    if not all_metrics:
        raise ValueError(
            "No configurations were evaluated. "
            "Provide a checkpoint or disable --skip-base-eval, or pass --base-eval-csv."
        )

    # Save results
    save_evaluation_results(all_results, all_metrics, mode, output_dir=target_output_dir)

    # Generate plots
    generate_evaluation_plots(all_metrics, mode, output_dir=target_output_dir)

    # Print summary
    print_evaluation_summary(all_metrics, mode)

    return {
        "results": all_results,
        "metrics": all_metrics,
        "output_dir": target_output_dir,
    }


def save_evaluation_results(
    all_results: Dict[str, List[Dict]],
    all_metrics: Dict[str, Dict],
    mode: str,
    output_dir: Optional[str] = None,
):
    """Save evaluation results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_output_dir = output_dir or EVALUATION_OUTPUT_DIR
    os.makedirs(target_output_dir, exist_ok=True)

    # Save detailed results to CSV
    for config_name, results in all_results.items():
        csv_path = os.path.join(target_output_dir, f"results_{config_name}_{timestamp}.csv")

        if mode == "nli":
            fieldnames = ["premise", "hypothesis", "true_label", "prediction", "raw_answer", "correct"]
        else:
            fieldnames = ["id", "question", "true_answer", "prediction", "raw_answer", "exact_match"]

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Saved {config_name} results to {csv_path}")

    # Save metrics to JSON
    metrics_path = os.path.join(target_output_dir, f"metrics_comparison_{timestamp}.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "timestamp": timestamp,
            "metrics": all_metrics,
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved comparison metrics to {metrics_path}")


def generate_evaluation_plots(
    all_metrics: Dict[str, Dict],
    mode: str,
    output_dir: Optional[str] = None,
):
    """Generate evaluation comparison plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_output_dir = output_dir or EVALUATION_OUTPUT_DIR
    plots_dir = os.path.join(target_output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metric_key = "accuracy" if mode == "nli" else "exact_match"

    # Plot 1: Accuracy/EM comparison
    comparison_data = {
        config_name: metrics[metric_key]
        for config_name, metrics in all_metrics.items()
    }

    plot_accuracy_comparison(
        comparison_data,
        mode=mode,
        save_path=os.path.join(plots_dir, f"accuracy_comparison_{timestamp}.png"),
    )

    perplexity_data = {
        config_name: metrics["wikitext_perplexity"]
        for config_name, metrics in all_metrics.items()
        if "wikitext_perplexity" in metrics
    }
    if perplexity_data:
        plot_perplexity_comparison(
            perplexity_data,
            save_path=os.path.join(plots_dir, f"wikitext_perplexity_comparison_{timestamp}.png"),
        )

    # For NLI mode: additional plots
    if mode == "nli":
        # Plot 2: Confusion matrices
        for config_name, metrics in all_metrics.items():
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                title=f"Confusion Matrix: {config_name}",
                save_path=os.path.join(plots_dir, f"confusion_matrix_{config_name}_{timestamp}.png"),
            )

        # Plot 3: F1 scores comparison
        f1_scores = {
            config_name: {
                label: metrics["per_label_metrics"][label]["f1"]
                for label in NLI_LABELS
            }
            for config_name, metrics in all_metrics.items()
        }

        plot_f1_scores_comparison(
            f1_scores,
            save_path=os.path.join(plots_dir, f"f1_comparison_{timestamp}.png"),
        )

    print(f"Saved evaluation plots to {plots_dir}")


def print_evaluation_summary(all_metrics: Dict[str, Dict], mode: str):
    """Print evaluation summary."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    metric_key = "accuracy" if mode == "nli" else "exact_match"
    metric_name = "Accuracy" if mode == "nli" else "Exact Match"

    print(f"\n{metric_name} Comparison:")
    print("-" * 80)

    for config_name, metrics in all_metrics.items():
        score = metrics[metric_key]
        count = metrics.get("correct_predictions", metrics.get("exact_match_count", 0))
        total = metrics["total_samples"]
        print(f"{config_name:25s}: {score:.4f} ({count}/{total})")

    ppl_configs = [
        (config_name, metrics["wikitext_perplexity"])
        for config_name, metrics in all_metrics.items()
        if "wikitext_perplexity" in metrics
    ]
    if ppl_configs:
        print("\nWikiText Perplexity Comparison:")
        print("-" * 80)
        for config_name, ppl in ppl_configs:
            print(f"{config_name:25s}: {ppl:.6f}")

    if mode == "nli":
        print(f"\nPer-Label F1 Scores:")
        print("-" * 80)
        print(f"{'Configuration':<25s} {'Entailment':<12s} {'Neutral':<12s} {'Contradiction':<12s}")
        print("-" * 80)

        for config_name, metrics in all_metrics.items():
            f1_scores = [
                metrics["per_label_metrics"][label]["f1"]
                for label in NLI_LABELS
            ]
            print(f"{config_name:<25s} {f1_scores[0]:<12.4f} {f1_scores[1]:<12.4f} {f1_scores[2]:<12.4f}")

    print("=" * 80 + "\n")
