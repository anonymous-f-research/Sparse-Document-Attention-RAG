# src/attack_pipeline/utils/ranked_list.py
from __future__ import annotations

import random
from typing import List, Sequence, Tuple, Union


def attack_config_requests_docs(pos_cfg: object) -> bool:
    """
    Check if an attack position configuration indicates malicious documents should be inserted.

    Args:
        pos_cfg: Attack position configuration. Can be:
                 - int: Non-zero values indicate documents should be inserted.
                 - list/tuple: Any non-zero element indicates documents should be inserted.
                 - other: Treated as no insertion requested.

    Returns:
        True if the configuration requests malicious document insertion, False otherwise.
    """
    if isinstance(pos_cfg, int):
        return pos_cfg != 0
    if isinstance(pos_cfg, (list, tuple)):
        for p in pos_cfg:
            if (p or 0) != 0:
                return True
        return False
    return False


def inject_malicious_docs_into_ranked_list(
    base_docs: List[str],
    malicious_docs: List[str],
    attack_pos: Union[int, Sequence[int]],
) -> List[str]:
    """
    Insert malicious documents into a ranked list at specified positions.

    Supports multiple insertion strategies:
    - Fixed position insertion (positive integers).
    - Random position insertion (negative integers or -1).
    - Mixed insertion using position lists.

    Args:
        base_docs: Original ranked list of documents.
        malicious_docs: List of malicious documents to insert.
        attack_pos: Position(s) for insertion:
                   - 0: No insertion (returns base_docs unchanged).
                   - Positive int: Insert at this rank position (1-indexed).
                   - -1: Insert at random positions.
                   - List/Tuple: Per-document insertion positions (parallel to malicious_docs).
                   - None values in list: Treated as special case.

    Returns:
        The ranked list with malicious documents injected at specified positions.

    Note:
        When attack_pos is a list shorter than malicious_docs, it's padded with -1 (random).
        Insertion maintains relative order of fixed positions but may use randomness for -1.
    """
    if not malicious_docs:
        return list(base_docs)

    ranked_docs = list(base_docs)

    if isinstance(attack_pos, int):
        if attack_pos == 0:
            return ranked_docs
        if attack_pos > 0:
            pos = max(0, min(attack_pos - 1, len(ranked_docs)))
            for md in malicious_docs:
                ranked_docs.insert(pos, md)
                pos += 1
            return ranked_docs
        if attack_pos == -1:
            for md in malicious_docs:
                idx = random.randint(0, len(ranked_docs))
                ranked_docs.insert(idx, md)
            return ranked_docs
        return ranked_docs

    pos_list = list(attack_pos)
    if len(pos_list) < len(malicious_docs):
        pos_list = pos_list + [-1] * (len(malicious_docs) - len(pos_list))
    elif len(pos_list) > len(malicious_docs):
        pos_list = pos_list[:len(malicious_docs)]

    fixed: List[Tuple[int, str]] = []
    specials: List[Tuple[int, str]] = []

    for md, p in zip(malicious_docs, pos_list):
        if p is None:
            specials.append((0, md))
        elif p > 0:
            fixed.append((p, md))
        else:
            specials.append((p, md))

    for p, md in sorted(fixed, key=lambda x: x[0], reverse=True):
        idx = max(0, min(p - 1, len(ranked_docs)))
        ranked_docs.insert(idx, md)

    for p, md in specials:
        if p == -1:
            idx = random.randint(0, len(ranked_docs))
            ranked_docs.insert(idx, md)
        else:
            continue

    return ranked_docs


def apply_ranked_list_order(ranked_docs: List[str], order_mode: str) -> List[str]:
    """
    Apply a ranking reordering policy to a list of documents.

    Args:
        ranked_docs: List of documents to reorder.
        order_mode: Reordering strategy:
                   - "bottom_up": Reverse the ranking order.
                   - "random": Randomly shuffle the documents.
                   - "top_down" or other: Return unchanged (default).

    Returns:
        The reordered document list according to the specified mode.

    Note:
        "bottom_up" preserves all documents but in reverse order.
        "random" uses Python's random.shuffle() for non-deterministic ordering.
        Unknown modes default to "top_down" (no reordering).
    """
    if order_mode == "bottom_up":
        return list(reversed(ranked_docs))
    if order_mode == "random":
        out = list(ranked_docs)
        random.shuffle(out)
        return out
    # "top_down" or unknown => no change
    return ranked_docs
