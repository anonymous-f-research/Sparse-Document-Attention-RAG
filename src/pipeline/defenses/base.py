from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

from src.pipeline.models.datamodels import DefenseOutput


class Defense(ABC):
    """
    Base defense interface.
    """

    @abstractmethod
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
        raise NotImplementedError
