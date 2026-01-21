# src/attack_pipeline/defenses/ragdefender_defense.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.pipeline.config import RAGDEFENDER_DEVICE, RAGDEFENDER_TASK, RANKER_MODEL_NAME
from src.pipeline.defenses.base import Defense
from src.pipeline.models.datamodels import DefenseOutput, make_mal_id


class RagDefenderDefense(Defense):
    """
    RAGDefender defense wrapper for filtering adversarial documents from RAG rankings.

    This defense uses the RAGDefender model to identify and remove documents that may
    be adversarial or irrelevant. It converts RAGDefender's output into kept/removed masks
    and reconstructs filtered rankings while optionally tracking malicious document survival.

    Attributes:
        _obj: Lazily initialized RAGDefender instance.
    """

    def __init__(self) -> None:
        """
        Initialize the RagDefenderDefense wrapper.

        The actual RAGDefender model is loaded lazily on first use via _init().
        """
        self._obj = None

    def _init(self) -> None:
        """
        Lazily initialize the RAGDefender model on first call.

        Imports RAGDefender and configures it with device and similarity model settings
        from the config module.

        Raises:
            RuntimeError: If RAGDefender cannot be imported or initialized.

        Note:
            Subsequent calls to this method are no-ops (caches the instance).
        """
        if self._obj is not None:
            return
        try:
            from ragdefender import RAGDefender
            self._obj = RAGDefender(device=RAGDEFENDER_DEVICE, similarity_model=RANKER_MODEL_NAME, gpu_id=RAGDEFENDER_DEVICE)
        except Exception as e:
            raise RuntimeError(
                "Failed to import/init RAGDefender. Verify installation and adjust import in RagDefenderDefense."
            ) from e

    @staticmethod
    def _build_keep_mask_by_text(original_docs: List[str], cleaned_docs: List[str]) -> List[bool]:
        """
        Build a binary mask indicating which original documents are in the cleaned set.

        Args:
            original_docs: Original document list (order preserved).
            cleaned_docs: Filtered document list (subset of original).

        Returns:
            List of booleans where mask[i] = True if original_docs[i] is in cleaned_docs.

        Note:
            - Uses simple membership checking (not multiset-safe)
            - Does not handle duplicate documents correctly
            - Case-sensitive comparison
        """
        # Your current version is a simple membership check (not multiset-safe).
        mask: List[bool] = []
        for d in original_docs:
            if d in cleaned_docs:
                mask.append(True)
            else:
                mask.append(False)
        return mask

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
        """
        Apply RAGDefender filtering to remove adversarial documents from rankings.

        Combines malicious and corpus documents, runs RAGDefender's defense, and
        reconstructs the output to separate corpus and malicious document results.

        Args:
            query_id: Unique identifier for this query.
            query: The original query text.
            corpus_docs: Retrieved documents from the RAG system.
            corpus_ids: Identifiers for corpus documents.
            corpus_scores: Relevance scores for corpus documents (optional).
            malicious_docs: Injected adversarial documents to filter.
            does_oracle: If True, track which malicious docs survive RAGDefender.
            persistent_cache: Unused (for interface compatibility).

        Returns:
            DefenseOutput containing:
                - ranked_docs: Corpus documents that passed RAGDefender filtering
                - ranked_ids: Corresponding document IDs
                - ranked_scores: Corresponding relevance scores
                - malicious_docs_survived: Malicious docs not filtered (if does_oracle=True)
                - doc_labels: None (RAGDefender doesn't provide per-doc labels)

        Note:
            - RAGDefender is initialized on first call (lazy initialization)
            - Malicious docs are presented first to RAGDefender for ranking
            - Filtering preserves original document order from joint ranking
        """
        self._init()

        # joint (malicious first)
        joint_docs: List[str] = []
        joint_ids: List[str] = []
        joint_scores: Optional[List[float]] = [] if corpus_scores is not None else None

        for i, md in enumerate(malicious_docs):
            joint_docs.append(md)
            joint_ids.append(make_mal_id(i))
            if joint_scores is not None:
                joint_scores.append(0.0)

        joint_docs.extend(corpus_docs)
        joint_ids.extend(corpus_ids)
        if joint_scores is not None:
            joint_scores.extend(corpus_scores)

        cleaned_docs = self._obj.defend(query=query, retrieved_docs=joint_docs, mode=RAGDEFENDER_TASK)
        keep_mask = self._build_keep_mask_by_text(joint_docs, cleaned_docs)

        kept_docs = [d for d, k in zip(joint_docs, keep_mask) if k]
        kept_ids = [i for i, k in zip(joint_ids, keep_mask) if k]
        kept_scores = [s for s, k in zip(joint_scores, keep_mask) if k] if joint_scores is not None else None

        kept_mals: List[str] = []
        kept_corpus_docs: List[str] = []
        kept_corpus_ids: List[str] = []
        kept_corpus_scores: Optional[List[float]] = [] if kept_scores is not None else None

        for idx, (d, did) in enumerate(zip(kept_docs, kept_ids)):
            if did.startswith("__MAL__") and does_oracle:
                kept_mals.append(d)
            else:
                kept_corpus_docs.append(d)
                kept_corpus_ids.append(did)
                if kept_corpus_scores is not None and kept_scores is not None and idx < len(kept_scores):
                    kept_corpus_scores.append(kept_scores[idx])

        return DefenseOutput(
            ranked_docs=kept_corpus_docs,
            ranked_ids=kept_corpus_ids,
            ranked_scores=kept_corpus_scores,
            malicious_docs_survived=kept_mals,
            doc_labels=None,
        )
