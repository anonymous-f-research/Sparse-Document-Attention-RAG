from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

from src.pipeline.models.datamodels import RetrievalBatch


class Retriever(ABC):
    """
    Abstract retriever interface.

    Implementations must return:
      RetrievalBatch(q_embs, docs_texts_full, ids_full, scores_full)
    """

    @abstractmethod
    def retrieve_batch(self, queries: Sequence[str], max_k_needed: int, embed_batch_size: int) -> RetrievalBatch:
        raise NotImplementedError
