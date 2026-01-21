from __future__ import annotations

import re
import string
import unicodedata


def normalize_answer(s: str) -> str:
    """Normalize answer for exact-match, identical to typical QA EM normalization."""
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


def extract_final_answer(text: str) -> str:
    """
    Best-effort post-processing to isolate the model's final answer.

    - Removes <think>...</think> blocks
    - Trims whitespace
    - Removes obvious prefixes like "Answer:" / "- Answer:"
    - Returns the first non-empty line if multiple lines exist
    """
    if text is None:
        return ""
    s = str(text)

    # Remove any chain-of-thought tags if present
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE)

    s = s.strip()

    # Remove common answer prefixes
    s = re.sub(r"^\s*(-\s*)?(final\s*answer\s*:|answer\s*:)\s*", "", s, flags=re.IGNORECASE).strip()

    # If multiple lines, take the first non-empty "answer-looking" line
    lines = [ln.strip() for ln in s.splitlines()]
    for ln in lines:
        if ln:
            return ln

    return ""


def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Returns True if normalized ground_truth is a substring of normalized prediction.
    (Keeps your original behavior.)
    """
    prediction_clean = re.sub(r"<think>.*?</think>", "", str(prediction), flags=re.DOTALL)
    return normalize_answer(ground_truth) in normalize_answer(prediction_clean)
