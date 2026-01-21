from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.pipeline.defenses.base import Defense
from src.pipeline.models.datamodels import DefenseOutput


class NoDefense(Defense):
    """No filtering; passthrough."""

    def apply(
        self,
        query_id: str,
        query: str,
        corpus_docs: List[str],
        corpus_ids: List[str],
        corpus_scores: Optional[List[float]],
        malicious_docs: List[str],
        does_oracle: bool,
        persistent_cache: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> DefenseOutput:
        # For oracle path, keep malicious docs separately exactly like your pipeline logic expects.
        return DefenseOutput(
            ranked_docs=list(corpus_docs),
            ranked_ids=list(corpus_ids),
            ranked_scores=list(corpus_scores) if corpus_scores is not None else None,
            malicious_docs_survived=list(malicious_docs) if does_oracle else [],
            doc_labels=None,
        )
