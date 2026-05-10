"""
NLI Sparse Attention Experiment Script

Compares the effect of sparse attention (SDAG) vs regular attention (CARG) on Natural Language Inference.
SDAG prevents the hypothesis from attending to the premise in the attention mechanism.
"""

from __future__ import annotations

import csv
import json
import os
import re
import string
import unicodedata
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local copies to keep this script runnable directly (without package imports).
SYSTEM_PROMPT_RAG = "You are a helpful assistant, below is a query from a user and some relevant contexts."
USER_RAG_PROMPT = """Answer the question concisely, based on the following passages.

passages:
{docs_text}

- Question: {query}

- Answer: """


def normalize_answer(s: str) -> str:
    """Normalize answer for exact-match."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return str(text).lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def qa_exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Returns True if normalized ground_truth is a substring of normalized prediction.
    Removes <think> blocks if they exist.
    """
    prediction = "" if prediction is None else str(prediction)
    ground_truth = "" if ground_truth is None else str(ground_truth)

    prediction_clean = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL)
    return normalize_answer(ground_truth) in normalize_answer(prediction_clean)

# =========================
# HYPERPARAMETERS (CONSTANTS)
# =========================

# Model Configuration
#MODEL_NAME = "Qwen/Qwen3.5-27B"
#MODEL_NAME = "microsoft/Phi-3-medium-4k-instruct"
# Optional automatic fallback when MODEL_NAME is unsupported by local transformers version.
# Example: FALLBACK_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
FALLBACK_MODEL_NAME = None
#MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" 
#MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # LLM model to use
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # Device for inference
MODEL_DTYPE = torch.bfloat16 if "cuda" in DEVICE else torch.float32  # Model precision

# Generation Parameters
TEMPERATURE = 0.1  # Temperature for sampling (lower = more deterministic)
TOP_P = 1.0  # Nucleus sampling parameter
MAX_NEW_TOKENS = 250  # Maximum tokens to generate per sample

# Dataset Parameters
TOTAL_SAMPLES = 3000  # Total samples to evaluate
SAMPLES_PER_LABEL = TOTAL_SAMPLES // 3  # Equal distribution across 3 labels
RANDOM_SEED = 42  # For reproducibility
EXPERIMENT_MODE = "qa"  # "nli" or "qa"
HOTPOTQA_CONFIG = "fullwiki"
HOTPOTQA_SPLIT = "train"
HOTPOTQA_QUESTION_TYPE = "bridge"
HOTPOTQA_MIN_SUPPORTING_DOCS = 2

# Batch Processing
BATCH_SIZE = 64  # Number of samples to process in parallel

# Parsing Configuration
USE_LLM_JUDGE = True  # Use LLM as judge for ambiguous answers (recommended)
USE_REASONING_PROMPT = True  # True: JSON reasoning response, False: direct single-label answer

# Debug Configuration (prompt split + attention mask inspection)
DEBUG_PRINT_SPLITS_AND_MASK = True
DEBUG_PRINT_MAX_SAMPLES = 1
DEBUG_MASK_MAX_TOKENS = 120  # Print at most this many rows/cols from the mask
_DEBUG_PRINTED_SAMPLES = 0
DEBUG_PRINT_EXAMPLE_IO = True
DEBUG_PRINT_EXAMPLE_MAX = 1
_DEBUG_PRINTED_EXAMPLES = 0

# Output Configuration
#OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/llama")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/mistral")
#OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/phi")
#OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/qwen")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")
PERFORMANCE_JSON = os.path.join(OUTPUT_DIR, "performance.json")

# Prompts
SYSTEM_PROMPT = """You are an NLP expert performing natural language inference on premise-hypothesis pairs.
Given a premise and a hypothesis, classify the hypothesis as:
- entailment
- neutral
- contradiction
with respect to the premise.
"""

USER_PROMPT_TEMPLATE_REASONING = """Premise: {premise}

Hypothesis: {hypothesis}

Determine the inference relation between the premise and the hypothesis.

Return your response as valid JSON with exactly these keys and order:
{{
  "explanation": "<concise reasoning in 1-2 sentences>",
  "answer": "<exactly one of: entailment, contradiction, neutral>"
}}
"""

USER_PROMPT_TEMPLATE_DIRECT = """Premise: {premise}
Hypothesis: {hypothesis}
Answer with exactly one of the options: entailment, contradiction, or neutral
Answer: """

USER_PROMPT_TEMPLATE = (
    USER_PROMPT_TEMPLATE_REASONING if USE_REASONING_PROMPT else USER_PROMPT_TEMPLATE_DIRECT
)
NLI_LABELS = ["entailment", "neutral", "contradiction"]


def get_user_prompt_template(use_reasoning_prompt: bool = USE_REASONING_PROMPT) -> str:
    """Select the user prompt template based on reasoning mode."""
    return USER_PROMPT_TEMPLATE_REASONING if use_reasoning_prompt else USER_PROMPT_TEMPLATE_DIRECT


# =========================
# DATA LOADING
# =========================


def load_balanced_snli(
    total_samples: int = TOTAL_SAMPLES,
    samples_per_label: int = SAMPLES_PER_LABEL,
    seed: int = RANDOM_SEED,
) -> List[Dict[str, str]]:
    """
    Load SNLI dataset and sample balanced examples across all labels.

    Args:
        total_samples: Total number of samples to return
        samples_per_label: Number of samples per label (entailment/neutral/contradiction)
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: premise, hypothesis, label
    """
    print("Loading SNLI dataset from HuggingFace...")
    dataset = load_dataset("stanfordnlp/snli", split="train")

    # Label mapping: 0=entailment, 1=neutral, 2=contradiction
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    # Group by label
    by_label = defaultdict(list)
    for item in dataset:
        if item["label"] in [0, 1, 2]:  # Filter out -1 (no gold label)
            by_label[item["label"]].append(item)

    # Sample from each label
    np.random.seed(seed)
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

    # Shuffle the balanced samples
    np.random.shuffle(balanced_samples)

    print(f"Loaded {len(balanced_samples)} balanced samples")
    label_counts = {label: sum(1 for s in balanced_samples if s["label"] == label) for label in ["entailment", "neutral", "contradiction"]}
    print(f"Label distribution: {label_counts}")

    return balanced_samples


def extract_supporting_fact_docs(example: Dict) -> List[str]:
    """
    Build deduplicated supporting-fact "documents" for HotpotQA from supporting_facts+context.

    Uses supporting-fact TITLES only (ignores sent_id). For each matched title,
    all sentences from that title's context entry are concatenated into one document:
        "<title>: <sent_0> <sent_1> ... <sent_n>"
    """
    supporting_facts = example.get("supporting_facts") or {}
    context = example.get("context") or {}

    sf_titles = list(supporting_facts.get("title", []))
    ctx_titles = list(context.get("title", []))
    ctx_sentences = list(context.get("sentences", []))

    title_to_sentences = {title: sents for title, sents in zip(ctx_titles, ctx_sentences)}

    docs: List[str] = []
    seen_titles = set()
    for title in sf_titles:
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)

        sents = title_to_sentences.get(title, [])
        if not sents:
            continue

        merged = " ".join(str(s).strip() for s in sents if str(s).strip())
        doc = f"{title}: {merged}".strip()
        if doc:
            docs.append(doc)

    return docs


def load_hotpotqa_bridge(
    total_samples: int = TOTAL_SAMPLES,
    seed: int = RANDOM_SEED,
    config_name: str = HOTPOTQA_CONFIG,
    split_name: str = HOTPOTQA_SPLIT,
    min_supporting_docs: int = HOTPOTQA_MIN_SUPPORTING_DOCS,
) -> List[Dict[str, str]]:
    """
    Load HotpotQA and sample bridge questions with deduplicated supporting-fact docs.
    """
    print(
        f"Loading HotpotQA ({config_name}/{split_name}) from HuggingFace "
        f"with min_supporting_docs={min_supporting_docs}..."
    )
    dataset = load_dataset("hotpot_qa", config_name, split=split_name)

    bridge_samples: List[Dict[str, str]] = []
    for item in dataset:
        if item.get("type") != HOTPOTQA_QUESTION_TYPE:
            continue
        docs = extract_supporting_fact_docs(item)
        if len(docs) < min_supporting_docs:
            continue
        bridge_samples.append({
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "supporting_docs": docs,
        })

    if not bridge_samples:
        raise ValueError(
            f"No HotpotQA bridge samples with at least {min_supporting_docs} supporting docs were found."
        )

    np.random.seed(seed)
    if len(bridge_samples) > total_samples:
        indices = np.random.choice(len(bridge_samples), total_samples, replace=False)
        sampled = [bridge_samples[i] for i in indices]
    else:
        sampled = bridge_samples
        print(f"Warning: only {len(sampled)} bridge samples available (requested {total_samples})")

    np.random.shuffle(sampled)
    print(f"Loaded {len(sampled)} HotpotQA bridge samples")
    return sampled


# =========================
# PROMPT CONSTRUCTION
# =========================


def build_nli_prompt(
    tokenizer,
    premise: str,
    hypothesis: str,
    system_prompt: str = SYSTEM_PROMPT,
    user_template: Optional[str] = None,
) -> Tuple[str, int, int, int, int, int]:
    """
    Build NLI prompt with chat template and compute token spans.

    Args:
        tokenizer: HuggingFace tokenizer with chat template
        premise: Premise text
        hypothesis: Hypothesis text
        system_prompt: System instruction
        user_template: User message template

    Returns:
        Tuple of:
            - chat_str: Formatted chat string
            - system_user_len: Token count before premise marker
            - premise_start: Token position where premise CONTENT starts (after "Premise: ")
            - premise_end: Token position where premise content ends
            - hypothesis_start: Token position where hypothesis CONTENT starts (after "Hypothesis: ")
            - hypothesis_end: Token position where hypothesis content ends
    """
    if user_template is None:
        user_template = get_user_prompt_template()

    user_content = user_template.format(premise=premise, hypothesis=hypothesis)

    chat_str = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Locate field markers in the rendered chat and derive content bounds by content length.
    # This is robust to multiline premise/hypothesis content.
    premise_marker = "Premise: "
    hypothesis_marker = "Hypothesis: "

    premise_marker_start = chat_str.find(premise_marker)
    if premise_marker_start == -1:
        raise ValueError("Could not find premise marker in chat string")
    premise_content_start_char = premise_marker_start + len(premise_marker)
    premise_content_end_char = premise_content_start_char + len(premise)

    hypothesis_marker_start = chat_str.find(hypothesis_marker, premise_content_end_char)
    if hypothesis_marker_start == -1:
        raise ValueError("Could not find hypothesis marker in chat string")
    hypothesis_content_start_char = hypothesis_marker_start + len(hypothesis_marker)
    hypothesis_content_end_char = hypothesis_content_start_char + len(hypothesis)

    if hypothesis_content_end_char > len(chat_str):
        raise ValueError("Computed hypothesis span exceeds chat string length")

    # Sanity-check exact extracted text against provided values.
    if chat_str[premise_content_start_char:premise_content_end_char] != premise:
        raise ValueError("Premise text span mismatch while computing token spans")
    if chat_str[hypothesis_content_start_char:hypothesis_content_end_char] != hypothesis:
        raise ValueError("Hypothesis text span mismatch while computing token spans")

    # Compute token spans
    before_premise_marker = chat_str[:premise_marker_start]
    system_user_len = len(tokenizer(before_premise_marker, return_tensors="pt").input_ids[0])

    premise_start = len(tokenizer(chat_str[:premise_content_start_char], return_tensors="pt").input_ids[0])
    premise_end = len(tokenizer(chat_str[:premise_content_end_char], return_tensors="pt").input_ids[0])
    hypothesis_start = len(tokenizer(chat_str[:hypothesis_content_start_char], return_tensors="pt").input_ids[0])
    hypothesis_end = len(tokenizer(chat_str[:hypothesis_content_end_char], return_tensors="pt").input_ids[0])

    return chat_str, system_user_len, premise_start, premise_end, hypothesis_start, hypothesis_end


def build_qa_prompt_and_spans(
    tokenizer,
    question: str,
    supporting_docs: List[str],
    system_prompt: str = SYSTEM_PROMPT_RAG,
    user_template: str = USER_RAG_PROMPT,
) -> Tuple[str, int, List[Tuple[int, int]], int, List[str]]:
    """
    Build QA prompt using RAG template and compute document spans for SDAG masking.

    Supporting facts are formatted as:
        doc 1: ...
        doc 2: ...
    """
    doc_lines = [f" doc {i}: {d}" for i, d in enumerate(supporting_docs, start=1) if d and d.strip()]
    docs_text = "\n".join(doc_lines)
    user_content = user_template.format(query=question, docs_text=docs_text)

    chat_str = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    doc_positions: List[int] = []
    search_from = 0
    for dl in doc_lines:
        pos = chat_str.find(dl, search_from)
        if pos == -1:
            raise ValueError("Could not locate a supporting-fact doc line in QA prompt")
        doc_positions.append(pos)
        search_from = pos + len(dl)

    question_marker = "- Question:"
    q_pos = chat_str.find(question_marker)
    if q_pos == -1:
        q_pos = len(chat_str)

    first_doc_pos = doc_positions[0] if doc_positions else q_pos
    sys_user_len = len(tokenizer(chat_str[:first_doc_pos], return_tensors="pt").input_ids[0])

    doc_token_spans: List[Tuple[int, int]] = []
    for dl, start_char in zip(doc_lines, doc_positions):
        start_tok = len(tokenizer(chat_str[:start_char], return_tensors="pt").input_ids[0])
        end_tok = len(tokenizer(chat_str[: start_char + len(dl)], return_tensors="pt").input_ids[0])
        doc_token_spans.append((start_tok, end_tok))

    qa_start = len(tokenizer(chat_str[:q_pos], return_tensors="pt").input_ids[0])

    return chat_str, sys_user_len, doc_token_spans, qa_start, doc_lines


# =========================
# ATTENTION MASKING
# =========================


def build_sdag_nli_mask(
    seq_len: int,
    system_user_len: int,
    premise_start: int,
    premise_end: int,
    hypothesis_start: int,
    hypothesis_end: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build SDAG attention mask for NLI.

    Behavior:
    - System/user prefix: standard causal
    - Premise marker tokens ("Premise: "): standard causal
    - Premise content tokens: standard causal
    - Hypothesis marker tokens ("Hypothesis: " and separators): standard causal
    - Hypothesis content tokens: sparse (cannot attend to premise content tokens)
    - QA/answer section (after hypothesis): standard causal

    Args:
        seq_len: Total sequence length
        system_user_len: Token count before premise marker
        premise_start: Token position where premise CONTENT starts (after marker)
        premise_end: Token position where premise content ends
        hypothesis_start: Token position where hypothesis CONTENT starts (after marker)
        hypothesis_end: Token position where hypothesis content ends
        device: Device for tensor

    Returns:
        Boolean mask tensor where True = can attend, False = blocked
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Basic span validation
    if not (0 <= system_user_len <= premise_start <= premise_end <= hypothesis_start <= hypothesis_end <= seq_len):
        raise ValueError(
            f"Invalid spans: sys_user={system_user_len}, premise_start={premise_start}, "
            f"premise_end={premise_end}, hypothesis_start={hypothesis_start}, "
            f"hypothesis_end={hypothesis_end}, seq_len={seq_len}"
        )

    # System/user section: standard causal.
    for i in range(system_user_len):
        mask[i, :i + 1] = True

    # Premise marker + premise content: regular causal.
    for i in range(system_user_len, premise_end):
        mask[i, :i + 1] = True

    # Between-premise-and-hypothesis marker section: regular causal.
    for i in range(premise_end, hypothesis_start):
        mask[i, :i + 1] = True

    # Hypothesis content: sparse.
    # Allow system/user + premise marker tokens.
    # Block premise content tokens.
    # Allow hypothesis marker section + hypothesis content causal prefix.
    for i in range(hypothesis_start, hypothesis_end):
        mask[i, :premise_start] = True
        mask[i, premise_start:premise_end] = False
        mask[i, premise_end:i + 1] = True

    # QA/answer + generated tokens: standard causal.
    for i in range(hypothesis_end, seq_len):
        mask[i, :i + 1] = True

    return mask


def build_sdag_qa_doc_mask(
    seq_len: int,
    sys_user_len: int,
    doc_token_spans: List[Tuple[int, int]],
    qa_start: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build SDAG-style QA mask where each supporting-fact document is isolated from others.

    Mirrors doc isolation logic from `src/sparse_attention_RAG/SDAG.py`:
    - System/user prefix: standard causal
    - Each doc token: attends to system/user + itself only
    - QA section and beyond: standard causal
    """
    if not (0 <= sys_user_len <= qa_start <= seq_len):
        raise ValueError(
            f"Invalid QA span bounds: sys_user_len={sys_user_len}, qa_start={qa_start}, seq_len={seq_len}"
        )

    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    for i in range(sys_user_len):
        mask[i, :i + 1] = True

    for d_start, d_end in doc_token_spans:
        if not (0 <= d_start <= d_end <= seq_len):
            raise ValueError(f"Invalid doc span: ({d_start}, {d_end}) for seq_len={seq_len}")
        for i in range(d_start, d_end):
            mask[i, :sys_user_len] = True
            mask[i, d_start:i + 1] = True

    for i in range(qa_start, seq_len):
        mask[i, :i + 1] = True

    return mask


def debug_print_prompt_splits_and_mask(
    tokenizer,
    chat_str: str,
    premise: str,
    hypothesis: str,
    system_user_len: int,
    premise_start: int,
    premise_end: int,
    hypothesis_start: int,
    hypothesis_end: int,
    mask: torch.Tensor,
    max_tokens: int = DEBUG_MASK_MAX_TOKENS,
) -> None:
    """
    Print prompt splits and a compact view of the final attention mask for debugging.
    """
    print("\n" + "=" * 80)
    print("DEBUG: NLI Prompt Splits + SDAG Mask")
    print("=" * 80)
    print("Premise:", repr(premise))
    print("Hypothesis:", repr(hypothesis))
    print(
        "Token spans:",
        f"sys_user_len={system_user_len},",
        f"premise=[{premise_start},{premise_end}),",
        f"hypothesis=[{hypothesis_start},{hypothesis_end})",
    )

    ids = tokenizer(chat_str, return_tensors="pt").input_ids[0]
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
    seq_len = len(tokens)
    n = min(seq_len, max_tokens)
    print(f"Sequence length: {seq_len} tokens (printing first {n})")

    def segment_of(idx: int) -> str:
        if idx < system_user_len:
            return "SYS_USER"
        if idx < premise_end:
            return "PREMISE"
        if idx < hypothesis_end:
            return "HYPOTHESIS"
        return "QA_ANS"

    print("\n-- Token Split Table --")
    for i in range(n):
        tok = tokens[i].replace("\n", "\\n")
        print(f"{i:04d}  {segment_of(i):10s}  {tok}")

    print("\n-- Mask (1=can attend, .=blocked) --")
    mask_cpu = mask[:n, :n].to("cpu")
    for r in range(n):
        row = "".join("1" if bool(x) else "." for x in mask_cpu[r])
        print(f"{r:04d} {row}")

    print("\n-- Segment Access Summary --")
    segment_ranges = {
        "SYS_USER": (0, min(system_user_len, seq_len)),
        "PREMISE": (min(system_user_len, seq_len), min(premise_end, seq_len)),
        "HYPOTHESIS": (min(premise_end, seq_len), min(hypothesis_end, seq_len)),
        "QA_ANS": (min(hypothesis_end, seq_len), seq_len),
    }
    for name, (start, end) in segment_ranges.items():
        if start >= end:
            print(f"{name:10s}: empty")
            continue
        seg_mask = mask[start:end, :].to("cpu")
        allowed = int(seg_mask.sum().item())
        total = seg_mask.numel()
        print(f"{name:10s}: allowed={allowed}/{total} ({allowed / max(total, 1):.4f})")
    print("=" * 80 + "\n")


def debug_print_example_io(
    mode: str,
    prompt: str,
    gt_answer: str,
    carg_answer: str,
    sdag_answer: str,
    premise: Optional[str] = None,
    hypothesis: Optional[str] = None,
    question: Optional[str] = None,
) -> None:
    """Print one full example: input, prompt, GT answer, and model outputs."""
    print("\n" + "=" * 80)
    print("DEBUG: Example IO")
    print("=" * 80)
    print(f"Mode: {mode}")
    if mode == "qa":
        print("Question:", question)
    else:
        print("Premise:", premise)
        print("Hypothesis:", hypothesis)
    print("\nPrompt:")
    print(prompt)
    print("\nGround Truth:", gt_answer)
    print("CARG Answer:", carg_answer)
    print("SDAG Answer:", sdag_answer)
    print("=" * 80 + "\n")


# =========================
# GENERATION
# =========================


def generate_with_custom_mask(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prompt_mask: Optional[torch.Tensor],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Generate text with optional custom attention mask.

    Args:
        model: Causal language model
        tokenizer: Tokenizer
        input_ids: Input token IDs [1, seq_len]
        prompt_mask: Optional custom attention mask [seq_len, seq_len], None for standard CARG
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text (decoded)
    """
    device = input_ids.device
    L0 = input_ids.size(1)

    model_dtype = next(model.parameters()).dtype
    NEG_INF = torch.finfo(model_dtype).min

    # Prepare attention mask
    if prompt_mask is not None:
        if prompt_mask.dtype == torch.bool:
            attn = torch.zeros_like(prompt_mask, dtype=model_dtype, device=device)
            attn = attn.masked_fill(~prompt_mask.to(device), NEG_INF)
        else:
            attn = prompt_mask.to(device, dtype=model_dtype)
        attn = attn.unsqueeze(0).unsqueeze(1)
    else:
        # Standard causal attention for CARG
        attn = None

    # First forward pass with custom mask
    with torch.no_grad():
        if attn is not None:
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        else:
            out = model(input_ids=input_ids, use_cache=True)

    past_key_values = out.past_key_values
    generated = input_ids

    # Sample first token
    logits = out.logits[:, -1, :]
    if temperature > 0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    generated = torch.cat([generated, next_token], dim=1)

    # Continue generation with standard causal masking
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(
                input_ids=generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = out.logits[:, -1, :]
        past_key_values = out.past_key_values

        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0][L0:], skip_special_tokens=True).strip()


# =========================
# ANSWER PARSING
# =========================

# LLM Judge prompt for ambiguous answers
LLM_JUDGE_PROMPT = """Given the following answer from an NLI classification task, determine which label it represents.

Answer: "{answer}"

The answer should represent one of these labels:
- entailment
- neutral
- contradiction

Respond with ONLY the label name (entailment, neutral, or contradiction). If the answer is unclear or doesn't match any label, respond with "unknown".

Label: """


def parse_nli_answer_with_llm(
    generated_text: str,
    model,
    tokenizer,
    device: str,
) -> str:
    """
    Use LLM as a judge to parse ambiguous NLI answers.

    Args:
        generated_text: Original generated answer
        model: LLM model for judgment
        tokenizer: Tokenizer
        device: Device for inference

    Returns:
        Parsed label: entailment, neutral, contradiction, or unknown
    """
    judge_prompt = LLM_JUDGE_PROMPT.format(answer=generated_text)

    # Format as chat
    chat_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": judge_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize and generate
    encoded = tokenizer(chat_str, return_tensors="pt").to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=10,
            temperature=0.0,  # Greedy for consistency
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    judge_response = tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip().lower()

    # Parse judge response (should be clean)
    if "entailment" in judge_response:
        return "entailment"
    elif "contradiction" in judge_response:
        return "contradiction"
    elif "neutral" in judge_response:
        return "neutral"
    else:
        return "unknown"


def parse_nli_answer(
    generated_text: str,
    model=None,
    tokenizer=None,
    device: str = "cpu",
    use_llm_judge: bool = USE_LLM_JUDGE,
    use_reasoning_prompt: bool = USE_REASONING_PROMPT,
) -> str:
    """
    Parse NLI label from generated text.

    First tries direct pattern matching. If no match found and use_llm_judge=True,
    uses LLM-as-a-judge for disambiguation.

    Args:
        generated_text: LLM generated response
        model: Optional LLM model for judge (required if use_llm_judge=True)
        tokenizer: Optional tokenizer for judge
        device: Device for LLM judge inference
        use_llm_judge: Whether to use LLM as judge for ambiguous answers
        use_reasoning_prompt: Whether outputs are expected in JSON reasoning format

    Returns:
        Predicted label: entailment, neutral, contradiction, or unknown
    """
    text_lower = generated_text.lower().strip()

    # Reasoning mode: prioritize JSON parsing first.
    if use_reasoning_prompt:
        # Try whole text, fenced JSON blocks, then generic JSON object extraction.
        json_candidates = [generated_text.strip()]

        fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", generated_text, flags=re.IGNORECASE)
        json_candidates.extend(block.strip() for block in fenced_blocks if block.strip())

        brace_match = re.search(r"\{[\s\S]*\}", generated_text)
        if brace_match:
            json_candidates.append(brace_match.group(0).strip())

        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, dict):
                for key in ("answer", "final_answer", "label"):
                    value = parsed.get(key)
                    if isinstance(value, str):
                        value_lower = value.strip().lower()
                        if value_lower in NLI_LABELS:
                            return value_lower

        # Malformed-JSON fallback: parse answer key by regex.
        json_answer_pattern = r'"(?:answer|final_answer|label)"\s*:\s*"?(entailment|neutral|contradiction)"?'
        match = re.search(json_answer_pattern, text_lower)
        if match:
            return match.group(1)

        # Backward compatibility with older reasoning format.
        final_answer_pattern = r'final\s*answer\s*:\s*(entailment|neutral|contradiction)\b'
        match = re.search(final_answer_pattern, text_lower)
        if match:
            return match.group(1)

    # Direct match patterns (most common, fastest)
    if text_lower in NLI_LABELS:
        return text_lower

    # Direct mode often returns "Answer: <label>".
    answer_line_pattern = r'^\s*answer\s*:\s*(entailment|neutral|contradiction)\s*$'
    match = re.search(answer_line_pattern, text_lower)
    if match:
        return match.group(1)

    # If no direct match and LLM judge is enabled and available, use it
    if use_llm_judge and model is not None and tokenizer is not None:
        return parse_nli_answer_with_llm(generated_text, model, tokenizer, device)

    # Check for exact word boundaries (fallback)
    pattern = r'\b(entailment|neutral|contradiction)\b'
    match = re.search(pattern, text_lower)
    if match:
        return match.group(1)

    # Final fallback: check for substring presence (no priority ordering)
    for label in NLI_LABELS:
        if label in text_lower:
            return label

    return "unknown"


# =========================
# BATCH PROCESSING
# =========================


def process_batch(
    model,
    tokenizer,
    batch_samples: List[Dict[str, str]],
    device: str,
) -> List[Dict[str, str]]:
    """
    Process a batch of NLI samples with both CARG and SDAG attention.
    CARG uses batched generation, SDAG must be sequential (custom masks).

    Args:
        model: Causal LLM
        tokenizer: Tokenizer
        batch_samples: List of samples with premise, hypothesis, label
        device: Device for computation

    Returns:
        List of result dicts with predictions and correctness
    """
    global _DEBUG_PRINTED_SAMPLES, _DEBUG_PRINTED_EXAMPLES

    # Build all prompts
    prompts = []
    metadata = []
    for sample in batch_samples:
        chat_str, sys_user_len, premise_start, premise_end, hypothesis_start, hypothesis_end = build_nli_prompt(
            tokenizer, sample["premise"], sample["hypothesis"]
        )
        prompts.append(chat_str)
        metadata.append({
            "premise": sample["premise"],
            "hypothesis": sample["hypothesis"],
            "true_label": sample["label"],
            "sys_user_len": sys_user_len,
            "premise_start": premise_start,
            "premise_end": premise_end,
            "hypothesis_start": hypothesis_start,
            "hypothesis_end": hypothesis_end,
        })

    # CARG: Batched generation with standard causal attention
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode CARG answers — slice new tokens by index (same pattern as SDAG / generate_with_custom_mask).
    # Decoding the full output and stripping the prompt via string comparison is unreliable because
    # special tokens are dropped and chat-template formatting may not round-trip cleanly.
    prompt_lengths = encoded["input_ids"].shape[1]  # all prompts padded to same length
    carg_answers = []
    for i, output in enumerate(outputs):
        # output includes the (padded) prompt tokens; skip them before decoding.
        new_tokens = output[prompt_lengths:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        carg_answers.append(answer)

    # SDAG: Must process individually (custom masks per sample)
    results = []
    for i, sample_meta in enumerate(metadata):
        # Tokenize individual sample for SDAG
        chat_str, sys_user_len, premise_start, premise_end, hypothesis_start, hypothesis_end = build_nli_prompt(
            tokenizer, sample_meta["premise"], sample_meta["hypothesis"]
        )
        encoded_single = tokenizer(chat_str, return_tensors="pt").to(device)
        input_ids = encoded_single["input_ids"]
        seq_len = input_ids.size(1)

        # SDAG: Sparse attention (hypothesis blocked from premise)
        sdag_mask = build_sdag_nli_mask(
            seq_len, sys_user_len, premise_start, premise_end, hypothesis_start, hypothesis_end, device=device
        )

        if DEBUG_PRINT_SPLITS_AND_MASK and _DEBUG_PRINTED_SAMPLES < DEBUG_PRINT_MAX_SAMPLES:
            debug_print_prompt_splits_and_mask(
                tokenizer=tokenizer,
                chat_str=chat_str,
                premise=sample_meta["premise"],
                hypothesis=sample_meta["hypothesis"],
                system_user_len=sys_user_len,
                premise_start=premise_start,
                premise_end=premise_end,
                hypothesis_start=hypothesis_start,
                hypothesis_end=hypothesis_end,
                mask=sdag_mask,
                max_tokens=DEBUG_MASK_MAX_TOKENS,
            )
            _DEBUG_PRINTED_SAMPLES += 1

        sdag_answer = generate_with_custom_mask(
            model, tokenizer, input_ids, prompt_mask=sdag_mask
        )

        # Parse both answers
        carg_pred = parse_nli_answer(
            carg_answers[i],
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_reasoning_prompt=USE_REASONING_PROMPT,
        )
        sdag_pred = parse_nli_answer(
            sdag_answer,
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_reasoning_prompt=USE_REASONING_PROMPT,
        )

        if DEBUG_PRINT_EXAMPLE_IO and _DEBUG_PRINTED_EXAMPLES < DEBUG_PRINT_EXAMPLE_MAX:
            debug_print_example_io(
                mode="nli",
                prompt=chat_str,
                gt_answer=sample_meta["true_label"],
                carg_answer=carg_answers[i],
                sdag_answer=sdag_answer,
                premise=sample_meta["premise"],
                hypothesis=sample_meta["hypothesis"],
            )
            _DEBUG_PRINTED_EXAMPLES += 1

        # Record results
        results.append({
            "premise": sample_meta["premise"],
            "hypothesis": sample_meta["hypothesis"],
            "true_label": sample_meta["true_label"],
            "carg_prediction": carg_pred,
            "sdag_prediction": sdag_pred,
            "carg_raw_answer": carg_answers[i],
            "sdag_raw_answer": sdag_answer,
            "carg_correct": int(carg_pred == sample_meta["true_label"]),
            "sdag_correct": int(sdag_pred == sample_meta["true_label"]),
        })

    return results


def clean_qa_answer(text: str) -> str:
    """Light cleanup for generated QA answers."""
    answer = (text or "").strip()
    answer = re.sub(r"^\s*assistant\s*:\s*", "", answer, flags=re.IGNORECASE)
    return answer.strip()


def process_batch_qa(
    model,
    tokenizer,
    batch_samples: List[Dict[str, str]],
    device: str,
) -> List[Dict[str, str]]:
    """
    Process a batch of QA samples with both CARG and SDAG doc-isolation attention.
    """
    global _DEBUG_PRINTED_EXAMPLES

    # Build prompts and span metadata
    prompts = []
    metadata = []
    for sample in batch_samples:
        chat_str, sys_user_len, doc_token_spans, qa_start, doc_lines = build_qa_prompt_and_spans(
            tokenizer=tokenizer,
            question=sample["question"],
            supporting_docs=sample["supporting_docs"],
        )
        prompts.append(chat_str)
        metadata.append({
            "id": sample.get("id", ""),
            "question": sample["question"],
            "answer": sample["answer"],
            "supporting_docs": sample["supporting_docs"],
            "sys_user_len": sys_user_len,
            "doc_token_spans": doc_token_spans,
            "qa_start": qa_start,
            "doc_lines": doc_lines,
        })

    # CARG: batched generation
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_len = encoded["input_ids"].shape[1]
    carg_answers = []
    for output in outputs:
        answer = tokenizer.decode(output[prompt_len:], skip_special_tokens=True).strip()
        carg_answers.append(clean_qa_answer(answer))

    # SDAG: sequential (custom doc-isolation masks per sample)
    results = []
    for i, sample_meta in enumerate(metadata):
        chat_str, sys_user_len, doc_token_spans, qa_start, _doc_lines = build_qa_prompt_and_spans(
            tokenizer=tokenizer,
            question=sample_meta["question"],
            supporting_docs=sample_meta["supporting_docs"],
        )
        encoded_single = tokenizer(chat_str, return_tensors="pt").to(device)
        input_ids = encoded_single["input_ids"]
        seq_len = input_ids.size(1)

        sdag_mask = build_sdag_qa_doc_mask(
            seq_len=seq_len,
            sys_user_len=sys_user_len,
            doc_token_spans=doc_token_spans,
            qa_start=qa_start,
            device=device,
        )
        sdag_answer = clean_qa_answer(
            generate_with_custom_mask(model, tokenizer, input_ids, prompt_mask=sdag_mask)
        )

        carg_answer = carg_answers[i]
        gold_answer = sample_meta["answer"]
        carg_em = int(qa_exact_match(carg_answer, gold_answer))
        sdag_em = int(qa_exact_match(sdag_answer, gold_answer))

        if DEBUG_PRINT_EXAMPLE_IO and _DEBUG_PRINTED_EXAMPLES < DEBUG_PRINT_EXAMPLE_MAX:
            debug_print_example_io(
                mode="qa",
                prompt=chat_str,
                gt_answer=gold_answer,
                carg_answer=carg_answer,
                sdag_answer=sdag_answer,
                question=sample_meta["question"],
            )
            _DEBUG_PRINTED_EXAMPLES += 1

        results.append({
            "id": sample_meta["id"],
            "question": sample_meta["question"],
            "true_answer": gold_answer,
            "supporting_docs": json.dumps(sample_meta["supporting_docs"], ensure_ascii=False),
            "carg_prediction": carg_answer,
            "sdag_prediction": sdag_answer,
            "carg_raw_answer": carg_answer,
            "sdag_raw_answer": sdag_answer,
            "carg_exact_match": carg_em,
            "sdag_exact_match": sdag_em,
        })

    return results


# =========================
# METRICS COMPUTATION
# =========================


def compute_metrics(results: List[Dict[str, str]], mode: str = EXPERIMENT_MODE) -> Dict:
    """
    Compute comprehensive metrics from results.

    Args:
        results: List of result dicts

    Returns:
        Dict with accuracy, per-label metrics, and confusion matrices
    """
    if mode == "qa":
        total = len(results)
        carg_correct = sum(r.get("carg_exact_match", 0) for r in results)
        sdag_correct = sum(r.get("sdag_exact_match", 0) for r in results)
        carg_em = carg_correct / total if total > 0 else 0.0
        sdag_em = sdag_correct / total if total > 0 else 0.0
        return {
            "CARG": {
                "exact_match": carg_em,
                "total_samples": total,
                "exact_match_count": carg_correct,
            },
            "SDAG": {
                "exact_match": sdag_em,
                "total_samples": total,
                "exact_match_count": sdag_correct,
            },
            "comparison": {
                "exact_match_difference": sdag_em - carg_em,
                "carg_exact_match": carg_em,
                "sdag_exact_match": sdag_em,
            },
        }

    labels = NLI_LABELS

    def calc_metrics_for_method(pred_key: str, correct_key: str):
        total = len(results)
        correct = sum(r[correct_key] for r in results)
        accuracy = correct / total if total > 0 else 0.0

        # Per-label metrics
        per_label = {}
        for label in labels:
            tp = sum(1 for r in results if r["true_label"] == label and r[pred_key] == label)
            fp = sum(1 for r in results if r["true_label"] != label and r[pred_key] == label)
            fn = sum(1 for r in results if r["true_label"] == label and r[pred_key] != label)
            tn = sum(1 for r in results if r["true_label"] != label and r[pred_key] != label)

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
        confusion = {true_l: {pred_l: 0 for pred_l in labels + ["unknown"]} for true_l in labels}
        for r in results:
            true_l = r["true_label"]
            pred_l = r[pred_key]
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

    carg_metrics = calc_metrics_for_method("carg_prediction", "carg_correct")
    sdag_metrics = calc_metrics_for_method("sdag_prediction", "sdag_correct")

    return {
        "CARG": carg_metrics,
        "SDAG": sdag_metrics,
        "comparison": {
            "accuracy_difference": sdag_metrics["accuracy"] - carg_metrics["accuracy"],
            "carg_accuracy": carg_metrics["accuracy"],
            "sdag_accuracy": sdag_metrics["accuracy"],
        }
    }


# =========================
# RESULTS SAVING
# =========================


def save_results(
    results: List[Dict[str, str]],
    metrics: Dict,
    model_name: str = MODEL_NAME,
    mode: str = EXPERIMENT_MODE,
):
    """
    Save results to CSV and metrics to JSON.

    Args:
        results: List of result dicts
        metrics: Computed metrics dict
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save CSV
    if mode == "qa":
        fieldnames = [
            "id",
            "question",
            "true_answer",
            "supporting_docs",
            "carg_prediction",
            "sdag_prediction",
            "carg_raw_answer",
            "sdag_raw_answer",
            "carg_exact_match",
            "sdag_exact_match",
        ]
    else:
        fieldnames = [
            "premise",
            "hypothesis",
            "true_label",
            "carg_prediction",
            "sdag_prediction",
            "carg_raw_answer",
            "sdag_raw_answer",
            "carg_correct",
            "sdag_correct",
        ]

    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_CSV}")

    # Save JSON metrics
    metrics_with_config = {
        "config": {
            "mode": mode,
            "model_name": model_name,
            "requested_model_name": MODEL_NAME,
            "device": DEVICE,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "total_samples": TOTAL_SAMPLES,
            "batch_size": BATCH_SIZE,
            "use_reasoning_prompt": USE_REASONING_PROMPT,
            "hotpotqa_config": HOTPOTQA_CONFIG if mode == "qa" else None,
            "hotpotqa_split": HOTPOTQA_SPLIT if mode == "qa" else None,
            "hotpotqa_question_type": HOTPOTQA_QUESTION_TYPE if mode == "qa" else None,
            "random_seed": RANDOM_SEED,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": metrics,
    }

    with open(PERFORMANCE_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics_with_config, f, indent=2)

    print(f"Performance metrics saved to: {PERFORMANCE_JSON}")


# =========================
# MAIN EXECUTION
# =========================


def main():
    """Main execution function."""
    if EXPERIMENT_MODE not in {"nli", "qa"}:
        raise ValueError(f"Invalid EXPERIMENT_MODE={EXPERIMENT_MODE!r}. Expected 'nli' or 'qa'.")

    print("=" * 80)
    print("Sparse Attention Experiment")
    print("=" * 80)
    print(f"Mode: {EXPERIMENT_MODE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Total samples: {TOTAL_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Temperature: {TEMPERATURE}")
    if EXPERIMENT_MODE == "nli":
        print(f"Use reasoning prompt: {USE_REASONING_PROMPT}")
    else:
        print(f"HotpotQA config/split/type: {HOTPOTQA_CONFIG}/{HOTPOTQA_SPLIT}/{HOTPOTQA_QUESTION_TYPE}")
    print("=" * 80)

    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load model and tokenizer
    def load_tokenizer_safe(model_name: str):
        try:
            tok = AutoTokenizer.from_pretrained(model_name)
        except AttributeError as e:
            # Gemma tokenizer configs may ship `extra_special_tokens` as a list, while
            # this transformers path expects a mapping. Retry with an explicit dict.
            if "has no attribute 'keys'" in str(e):
                print("Tokenizer compatibility fallback: retrying with extra_special_tokens={}")
                tok = AutoTokenizer.from_pretrained(model_name, extra_special_tokens={})
            else:
                raise

        tok.padding_side = "left"  # Required for decoder-only models with batched generation
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    active_model_name = MODEL_NAME
    print(f"\nLoading model: {active_model_name}...")
    tokenizer = load_tokenizer_safe(active_model_name)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            active_model_name,
            torch_dtype=MODEL_DTYPE,
            device_map=DEVICE,
        )
    except ValueError as e:
        is_unknown_arch = "does not recognize this architecture" in str(e)
        is_gemma4 = "model type `gemma4`" in str(e)
        if is_unknown_arch and is_gemma4:
            if FALLBACK_MODEL_NAME:
                print(f"Model '{active_model_name}' unsupported in local transformers. Falling back to '{FALLBACK_MODEL_NAME}'.")
                active_model_name = FALLBACK_MODEL_NAME
                tokenizer = load_tokenizer_safe(active_model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    active_model_name,
                    torch_dtype=MODEL_DTYPE,
                    device_map=DEVICE,
                )
            else:
                raise RuntimeError(
                    "Model architecture 'gemma4' is not supported by your local setup "
                    "(Python 3.9 + transformers 4.57.x). "
                    "Use a supported model (e.g., Qwen/Llama) or run with Python>=3.10 "
                    "and a newer transformers build that includes gemma4."
                ) from e
        else:
            raise

    model.eval()
    # Ensure generate() always has a valid padding token for decoder-only models.
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    print(f"Model loaded successfully: {active_model_name}")

    # Load data
    if EXPERIMENT_MODE == "nli":
        samples = load_balanced_snli()
    else:
        samples = load_hotpotqa_bridge()

    # Process in batches
    all_results = []
    num_batches = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE

    carg_running_correct = 0
    sdag_running_correct = 0
    total_processed = 0

    print(f"\nProcessing {len(samples)} samples in {num_batches} batches...")

    with tqdm(total=len(samples), desc="Processing", unit="sample") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(samples))
            batch = samples[start_idx:end_idx]

            if EXPERIMENT_MODE == "nli":
                batch_results = process_batch(model, tokenizer, batch, DEVICE)
            else:
                batch_results = process_batch_qa(model, tokenizer, batch, DEVICE)
            all_results.extend(batch_results)

            # Update running statistics
            if EXPERIMENT_MODE == "nli":
                carg_running_correct += sum(r["carg_correct"] for r in batch_results)
                sdag_running_correct += sum(r["sdag_correct"] for r in batch_results)
            else:
                carg_running_correct += sum(r["carg_exact_match"] for r in batch_results)
                sdag_running_correct += sum(r["sdag_exact_match"] for r in batch_results)
            total_processed += len(batch_results)

            carg_acc = carg_running_correct / total_processed
            sdag_acc = sdag_running_correct / total_processed

            if EXPERIMENT_MODE == "nli":
                pbar.set_postfix({
                    "CARG_Acc": f"{carg_acc:.3f}",
                    "SDAG_Acc": f"{sdag_acc:.3f}",
                })
            else:
                pbar.set_postfix({
                    "CARG_EM": f"{carg_acc:.3f}",
                    "SDAG_EM": f"{sdag_acc:.3f}",
                })
            pbar.update(len(batch_results))

    # Compute final metrics
    print("\nComputing final metrics...")
    metrics = compute_metrics(all_results, mode=EXPERIMENT_MODE)

    # Save results
    save_results(all_results, metrics, model_name=active_model_name, mode=EXPERIMENT_MODE)

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total samples: {metrics['CARG']['total_samples']}")
    if EXPERIMENT_MODE == "nli":
        print(f"\nCARG (Regular Attention):")
        print(f"  Accuracy: {metrics['CARG']['accuracy']:.4f} ({metrics['CARG']['correct_predictions']}/{metrics['CARG']['total_samples']})")
        print(f"\nSDAG (Sparse Attention - Hypothesis blocked from Premise):")
        print(f"  Accuracy: {metrics['SDAG']['accuracy']:.4f} ({metrics['SDAG']['correct_predictions']}/{metrics['SDAG']['total_samples']})")
        print(f"\nDifference (SDAG - CARG): {metrics['comparison']['accuracy_difference']:+.4f}")
        print("=" * 80)

        print("\nPer-label F1 Scores:")
        print(f"{'Label':<15} {'CARG F1':<10} {'SDAG F1':<10} {'Difference':<10}")
        print("-" * 50)
        for label in NLI_LABELS:
            carg_f1 = metrics['CARG']['per_label_metrics'][label]['f1']
            sdag_f1 = metrics['SDAG']['per_label_metrics'][label]['f1']
            diff = sdag_f1 - carg_f1
            print(f"{label:<15} {carg_f1:<10.4f} {sdag_f1:<10.4f} {diff:+<10.4f}")
    else:
        print(f"\nCARG (Regular Attention):")
        print(f"  Exact Match: {metrics['CARG']['exact_match']:.4f} ({metrics['CARG']['exact_match_count']}/{metrics['CARG']['total_samples']})")
        print(f"\nSDAG (Sparse Attention - Supporting facts isolated):")
        print(f"  Exact Match: {metrics['SDAG']['exact_match']:.4f} ({metrics['SDAG']['exact_match_count']}/{metrics['SDAG']['total_samples']})")
        print(f"\nDifference (SDAG - CARG): {metrics['comparison']['exact_match_difference']:+.4f}")

    print("\n" + "=" * 80)
    print("Experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
