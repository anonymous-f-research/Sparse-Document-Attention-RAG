from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PairSpec:
    """Experiment condition identified by (top_k, attacker_position)."""
    top_k: int
    attacker_pos: int


@dataclass
class QueryData:
    """
    Unified dataset input.

    - short_answers: list-of-list (each query may have multiple GT strings).
    - false_answer_groups / malicious_doc_groups: optional presets (CSV mode).
    """
    query_ids: List[str]
    questions: List[str]
    short_answers: List[List[str]]
    false_answer_groups: Optional[List[List[str]]]
    malicious_doc_groups: Optional[List[List[str]]]


@dataclass
class RetrievalBatch:
    """
    Retrieval output for a batch.

    docs_texts_full / ids_full / scores_full are aligned:
      outer list: per query
      inner list: ranked docs length == max_k_needed
    """
    q_embs: List[Any]
    docs_texts_full: List[List[str]]
    ids_full: List[List[str]]
    scores_full: List[List[float]]


@dataclass
class Resources:
    """Heavy objects initialized once and reused."""
    ranker: Any
    tokenizer: Any
    llm_model: Any
    dense_index: Any
    dense_meta: Any
    sparse_searcher: Any


@dataclass
class DefenseOutput:
    """
    Mirrors your DefenseOutput usage.

    ranked_docs / ranked_ids / ranked_scores = filtered corpus-side list
    malicious_docs_survived = malicious docs that survived defense (oracle path)
    doc_labels = optional labels-by-id (discern)
    """
    ranked_docs: List[str]
    ranked_ids: List[str]
    ranked_scores: Optional[List[float]]
    malicious_docs_survived: List[str]
    doc_labels: Optional[Dict[str, str]]


def make_mal_id(i: int) -> str:
    """Matches your __MAL__ id convention."""
    return f"__MAL__{i}"
