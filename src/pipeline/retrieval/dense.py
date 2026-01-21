from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.pipeline.config import RANKER_MODEL_NAME
from src.pipeline.models.datamodels import RetrievalBatch
from src.pipeline.retrieval.retriever import Retriever


def load_faiss_index_and_meta(index_path: str, meta_path: str) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Load a FAISS index and its associated metadata from disk.

    Args:
        index_path: Path to the FAISS index file.
        meta_path: Path to the metadata JSONL file containing document metadata.

    Returns:
        A tuple containing:
            - The loaded FAISS index object
            - A list of metadata dictionaries (one per document)

    Raises:
        FileNotFoundError: If either the index or metadata file does not exist.
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata jsonl not found at {meta_path}")

    index = faiss.read_index(index_path)
    meta: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return index, meta


def embed_queries(ranker: SentenceTransformer, queries: List[str], batch_size: int) -> np.ndarray:
    """
    Generate dense embeddings for a list of queries using a SentenceTransformer model.

    Args:
        ranker: SentenceTransformer model used to encode queries.
        queries: List of query strings to embed.
        batch_size: Number of queries to process in each batch.

    Returns:
        A numpy array of shape (len(queries), embedding_dim) containing normalized embeddings.

    Note:
        If the model name contains 'e5', queries are prefixed with "query: " as required by E5 models.
    """
    prefixed = ["query: " + q if "e5" in RANKER_MODEL_NAME.lower() else q for q in queries]
    all_embs = []
    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i:i + batch_size]
        emb = ranker.encode(batch, convert_to_tensor=True, batch_size=batch_size, normalize_embeddings=True)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


def search_index(index: Any, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index for the top-k nearest neighbors of query embeddings.

    Args:
        index: The FAISS index object to search.
        query_embeddings: Array of query embeddings with shape (num_queries, embedding_dim).
        top_k: Number of nearest neighbors to retrieve for each query.

    Returns:
        A tuple containing:
            - indices: Array of shape (num_queries, top_k) with document indices
            - scores: Array of shape (num_queries, top_k) with similarity scores
    """
    q = query_embeddings.astype(np.float32)
    scores, indices = index.search(q, top_k)
    return indices, scores


def materialize_dense_hits(
    indices: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    meta: Sequence[Dict[str, Any]],
) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Convert raw search results into materialized document texts, IDs, and scores.

    Args:
        indices: 2D sequence of document indices from search results.
        scores: 2D sequence of similarity scores corresponding to the indices.
        meta: List of metadata dictionaries indexed by document ID.

    Returns:
        A tuple containing:
            - docs_texts_batch: List of lists containing document texts
            - ids_batch: List of lists containing document IDs
            - scores_batch: List of lists containing similarity scores

    Note:
        Invalid indices are handled gracefully by returning empty strings and "NA" values.
    """
    docs_texts_batch: List[List[str]] = []
    ids_batch: List[List[str]] = []
    scores_batch: List[List[float]] = []

    for row_indices, row_scores in zip(indices, scores):
        row_texts: List[str] = []
        row_ids: List[str] = []
        row_sc: List[float] = []
        for idx, sc in zip(row_indices, row_scores):
            if 0 <= idx < len(meta):
                row_texts.append(meta[idx].get("text", ""))
                row_ids.append(f"{meta[idx].get('id', 'NA')}")
                row_sc.append(float(sc))
            else:
                row_texts.append("")
                row_ids.append("NA")
                row_sc.append(float(sc))
        docs_texts_batch.append(row_texts)
        ids_batch.append(row_ids)
        scores_batch.append(row_sc)

    return docs_texts_batch, ids_batch, scores_batch


class DenseRetriever(Retriever):
    """
    Dense retrieval system using FAISS indexing and E5 sentence embeddings.

    This retriever encodes queries into dense vectors and searches a pre-built FAISS index
    to find the most similar documents.

    Attributes:
        ranker: SentenceTransformer model for encoding queries and documents.
        index: FAISS index for efficient similarity search.
        meta: Metadata for all indexed documents.
    """

    def __init__(self, ranker: SentenceTransformer, index: Any, meta: List[Dict[str, Any]]) -> None:
        """
        Initialize the DenseRetriever.

        Args:
            ranker: SentenceTransformer model for generating embeddings.
            index: Pre-built FAISS index containing document embeddings.
            meta: List of metadata dictionaries for documents in the index.
        """
        self.ranker = ranker
        self.index = index
        self.meta = meta

    def retrieve_batch(self, queries: Sequence[str], max_k_needed: int, embed_batch_size: int) -> RetrievalBatch:
        """
        Retrieve the top-k documents for a batch of queries.

        Args:
            queries: Sequence of query strings to process.
            max_k_needed: Maximum number of documents to retrieve per query.
            embed_batch_size: Batch size for embedding generation.

        Returns:
            A RetrievalBatch object containing:
                - Query embeddings
                - Full batch of retrieved document texts
                - Full batch of document IDs
                - Full batch of similarity scores
        """
        q_embs = embed_queries(self.ranker, list(queries), batch_size=embed_batch_size)
        indices, scores = search_index(self.index, q_embs, top_k=max_k_needed)
        docs_texts, doc_ids, doc_scores = materialize_dense_hits(indices, scores, self.meta)
        return RetrievalBatch(q_embs=list(q_embs), docs_texts_full=docs_texts, ids_full=doc_ids, scores_full=doc_scores)
