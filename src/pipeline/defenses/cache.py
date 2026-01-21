from __future__ import annotations

import json
import os
from typing import Dict, Tuple


def load_discern_labels_jsonl(path: str) -> Dict[Tuple[str, str], str]:
    """
    Load Discern classification labels from a JSONL file into an in-memory cache.

    Each line in the JSONL file should be a JSON object with keys:
    - query_id: The query identifier
    - doc_id: The document identifier
    - label: Classification label ("clean" or "perturbed")

    Args:
        path: File path to the JSONL labels file.

    Returns:
        Dictionary mapping (query_id, doc_id) tuples to label strings ("clean" or "perturbed").
        Returns empty dictionary if path is empty, doesn't exist, or file is malformed.

    Side Effects:
        Prints debug messages to stdout indicating loaded count and file path.

    Note:
        - Invalid JSON lines are silently skipped
        - Only entries with both query_id and doc_id are cached
        - Label values are normalized to lowercase
        - Labels other than "clean" or "perturbed" are skipped
    """
    cache: Dict[Tuple[str, str], str] = {}
    if not path:
        return cache
    if not os.path.exists(path):
        print(f"[Discern] labels load path not found: {path}")
        return cache

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                qid = str(obj.get("query_id", "")).strip()
                did = str(obj.get("doc_id", "")).strip()
                lab = str(obj.get("label", "")).strip().lower()
                if (qid and did) and (lab in ("clean", "perturbed")):
                    cache[(qid, did)] = lab
            except Exception:
                continue

    print(f"[Discern] Loaded {len(cache)} labels from {path}")
    return cache


def save_discern_labels_jsonl(path: str, cache: Dict[Tuple[str, str], str]) -> None:
    """
    Save Discern classification labels from cache to a JSONL file.

    Each (query_id, doc_id) -> label mapping is written as one JSON line.

    Args:
        path: File path where the JSONL labels file will be written.
        cache: Dictionary mapping (query_id, doc_id) tuples to label strings.

    Returns:
        None. Output is written to file at path.

    Side Effects:
        - Creates parent directories if they don't exist
        - Creates or overwrites file at path
        - Prints debug message to stdout with count and path

    Note:
        - Uses UTF-8 encoding
        - Sets ensure_ascii=False to preserve non-ASCII characters
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for (qid, did), lab in cache.items():
            f.write(json.dumps({"query_id": qid, "doc_id": did, "label": lab}, ensure_ascii=False) + "\n")
    print(f"[Discern] Saved {len(cache)} labels to {path}")
