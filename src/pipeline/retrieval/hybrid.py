from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from src.pipeline.models.datamodels import RetrievalBatch
from src.pipeline.retrieval.retriever import Retriever


def split_k_between_sparse_and_dense(top_k: int, rng: random.Random) -> Tuple[int, int]:
    """
    Split the desired number of results between sparse and dense retrievers.

    Args:
        top_k: Total number of results needed.
        rng: Random number generator for tie-breaking when top_k is odd.

    Returns:
        A tuple of (k_sparse, k_dense) where k_sparse + k_dense = top_k.
        When top_k is odd, one retriever gets one extra result randomly.
    """
    k_half = top_k // 2
    if (top_k % 2) == 0:
        return k_half, k_half
    if rng.random() < 0.5:
        return k_half + 1, k_half
    return k_half, k_half + 1


def rrf_fuse_one_query(
    sparse_docs: List[str],
    sparse_ids: List[str],
    dense_docs: List[str],
    dense_ids: List[str],
    k0: int = 60,
) -> Tuple[List[str], List[str], List[float]]:
    """
    Fuse results from sparse and dense retrievers using Reciprocal Rank Fusion (RRF).

    RRF combines rankings from multiple retrievers by computing a reciprocal rank score
    for each document: score = 1/(k0 + rank). Documents appearing in both rankings
    contribute scores from both sources.

    Args:
        sparse_docs: List of document texts from sparse retriever.
        sparse_ids: List of document IDs from sparse retriever (parallel to sparse_docs).
        dense_docs: List of document texts from dense retriever.
        dense_ids: List of document IDs from dense retriever (parallel to dense_docs).
        k0: Smoothing parameter for RRF (default 60). Higher values reduce the impact of rank differences.

    Returns:
        A tuple containing:
            - fused_docs: Merged document texts sorted by fusion score (descending).
            - fused_ids: Corresponding document IDs in the same order.
            - fused_scores: RRF scores for each document.

    Note:
        Documents are deduplicated by ID (or text if ID is missing/invalid).
        Invalid IDs ("NA" or empty strings) fall back to using document text as the key.
    """
    def key_for(doc_id: str, doc_text: str) -> str:
        if doc_id is not None and doc_id != "" and doc_id != "NA":
            return doc_id
        return doc_text

    sparse_rank = {}
    for i, (d, did) in enumerate(zip(sparse_docs, sparse_ids), start=1):
        sparse_rank[key_for(did, d)] = i

    dense_rank = {}
    for i, (d, did) in enumerate(zip(dense_docs, dense_ids), start=1):
        dense_rank[key_for(did, d)] = i

    all_keys = set(sparse_rank.keys()) | set(dense_rank.keys())

    rep_doc = {}
    rep_id = {}

    for d, did in zip(sparse_docs, sparse_ids):
        k = key_for(did, d)
        if k not in rep_doc:
            rep_doc[k] = d
            rep_id[k] = did

    for d, did in zip(dense_docs, dense_ids):
        k = key_for(did, d)
        if k not in rep_doc:
            rep_doc[k] = d
            rep_id[k] = did

    fused = []
    for k in all_keys:
        score = 0.0
        if k in sparse_rank:
            score += 1.0 / float(k0 + sparse_rank[k])
        if k in dense_rank:
            score += 1.0 / float(k0 + dense_rank[k])
        fused.append((score, rep_doc[k], rep_id[k]))

    fused.sort(key=lambda x: x[0], reverse=True)
    fused_scores = [s for (s, _, _) in fused]
    fused_docs = [d for (_, d, _) in fused]
    fused_ids = [did for (_, _, did) in fused]
    return fused_docs, fused_ids, fused_scores


def fuse_sparse_and_dense_batch(
    sparse_texts: List[List[str]],
    sparse_ids: List[List[str]],
    dense_texts: List[List[str]],
    dense_ids: List[List[str]],
    top_k: int,
    seed: int,
    k0: int = 60,
) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Fuse results from sparse and dense retrievers for a batch of queries using RRF.

    This function processes multiple queries, splits the top_k budget between
    sparse and dense results, and fuses them using Reciprocal Rank Fusion.

    Args:
        sparse_texts: List of document text lists from sparse retriever (one list per query).
        sparse_ids: List of document ID lists from sparse retriever (one list per query).
        dense_texts: List of document text lists from dense retriever (one list per query).
        dense_ids: List of document ID lists from dense retriever (one list per query).
        top_k: Maximum number of fused results per query.
        seed: Random seed for reproducible k-splitting between sparse and dense.
        k0: RRF smoothing parameter (default 60).

    Returns:
        A tuple containing:
            - fused_texts_batch: List of fused document text lists (one per query).
            - fused_ids_batch: List of fused document ID lists (one per query).
            - fused_scores_batch: List of fused score lists (one per query).
    """
    rng = random.Random(seed)

    fused_texts_batch = []
    fused_ids_batch = []
    fused_scores_batch = []

    for s_docs_full, s_ids_full, d_docs_full, d_ids_full in zip(sparse_texts, sparse_ids, dense_texts, dense_ids):
        k_sparse, k_dense = split_k_between_sparse_and_dense(top_k, rng)

        s_docs = s_docs_full[:k_sparse]
        s_ids = s_ids_full[:k_sparse]
        d_docs = d_docs_full[:k_dense]
        d_ids = d_ids_full[:k_dense]

        fused_docs, fused_doc_ids, fused_scores = rrf_fuse_one_query(
            sparse_docs=s_docs,
            sparse_ids=s_ids,
            dense_docs=d_docs,
            dense_ids=d_ids,
            k0=k0,
        )

        fused_texts_batch.append(fused_docs[:top_k])
        fused_ids_batch.append(fused_doc_ids[:top_k])
        fused_scores_batch.append(fused_scores[:top_k])

    return fused_texts_batch, fused_ids_batch, fused_scores_batch


class HybridRetriever(Retriever):
    """
    Hybrid retrieval system combining sparse and dense retrievers with RRF fusion.

    This retriever runs both sparse (BM25) and dense (embedding-based) retrieval
    in parallel and fuses their results using Reciprocal Rank Fusion to achieve
    better coverage and relevance.

    Attributes:
        dense: Dense retriever instance (embedding-based).
        sparse: Sparse retriever instance (BM25-based).
        seed: Random seed for reproducible k-splitting.
        k0: RRF smoothing parameter.
    """

    def __init__(self, dense_retriever: Retriever, sparse_retriever: Retriever, seed: int, k0: int = 60) -> None:
        """
        Initialize the HybridRetriever.

        Args:
            dense_retriever: Dense retriever instance for embedding-based search.
            sparse_retriever: Sparse retriever instance for BM25-based search.
            seed: Random seed for reproducible behavior in k-splitting.
            k0: RRF smoothing parameter (default 60).
        """
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.seed = seed
        self.k0 = k0

    def retrieve_batch(self, queries: Sequence[str], max_k_needed: int, embed_batch_size: int) -> RetrievalBatch:
        """
        Retrieve and fuse results from both dense and sparse retrievers.

        Args:
            queries: Sequence of query strings to process.
            max_k_needed: Maximum number of fused results per query.
            embed_batch_size: Batch size for embedding generation (passed to dense retriever).

        Returns:
            A RetrievalBatch object containing:
                - Query embeddings from the dense retriever.
                - Fused document texts from both retrievers.
                - Fused document IDs from both retrievers.
                - Fused similarity scores.
        """
        dense = self.dense.retrieve_batch(queries, max_k_needed, embed_batch_size)
        sparse = self.sparse.retrieve_batch(queries, max_k_needed, embed_batch_size)

        fused_texts, fused_ids, fused_scores = fuse_sparse_and_dense_batch(
            sparse_texts=sparse.docs_texts_full,
            sparse_ids=sparse.ids_full,
            dense_texts=dense.docs_texts_full,
            dense_ids=dense.ids_full,
            top_k=max_k_needed,
            seed=self.seed,
            k0=self.k0,
        )
        # q_embs from dense (keeps your original usage)
        return RetrievalBatch(q_embs=dense.q_embs, docs_texts_full=fused_texts, ids_full=fused_ids, scores_full=fused_scores)
