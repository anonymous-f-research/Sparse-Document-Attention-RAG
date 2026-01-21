from __future__ import annotations

import random
import re
from typing import List


def doc_contains_any_gt(doc: str, gt_answers: List[str]) -> bool:
    """
    True if the doc contains (case-insensitive) any GT string as a substring.
    """
    if not doc:
        return False
    d = doc.lower()
    for gt in gt_answers:
        if gt and gt.strip() and gt.strip().lower() in d:
            return True
    return False


def replace_gt_with_false(doc: str, gt_answers: List[str], false_answer: str) -> str:
    """
    Replace occurrences of any GT answer string with the false answer (case-insensitive).
    Conservative implementation: substring replacement using regex escaping.
    """
    if not doc:
        return ""
    if not false_answer:
        return doc

    out = doc
    for gt in gt_answers:
        if not gt or not gt.strip():
            continue
        pattern = re.escape(gt.strip())
        out = re.sub(pattern, false_answer, out, flags=re.IGNORECASE)
    return out


def build_docs_for_attack(
    docs: List[str],
    attacked_idx: int,
    attack_pos: int,
    top_k: int,
) -> List[str]:
    """
    Reorder docs so docs[attacked_idx] is moved to attacker position, then truncate to top_k.
    attack_pos semantics:
      - 0: no move (keep as is)
      - >0: 1-indexed insertion position
      - -1: random insertion
    """
    if not docs:
        return []

    attacked_idx = max(0, min(attacked_idx, len(docs) - 1))

    out = list(docs)
    attacked_doc = out.pop(attacked_idx)

    if attack_pos == 0:
        # Put it back in its original place (no-op)
        out.insert(attacked_idx, attacked_doc)
        return out[:top_k]

    if attack_pos == -1:
        insert_at = random.randint(0, len(out))
        out.insert(insert_at, attacked_doc)
        return out[:top_k]

    # attack_pos > 0 (1-indexed)
    insert_at = max(0, min(int(attack_pos) - 1, len(out)))
    out.insert(insert_at, attacked_doc)
    return out[:top_k]
