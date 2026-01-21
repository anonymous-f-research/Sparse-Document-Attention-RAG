from __future__ import annotations

import random
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.pipeline.config import RANKER_MODEL_NAME


def encode_texts_with_ranker(ranker: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Encode a list of text strings into dense embeddings using a SentenceTransformer model.

    Args:
        ranker: SentenceTransformer model for encoding.
        texts: List of text strings to embed.

    Returns:
        Numpy array of shape (len(texts), embedding_dim) containing normalized embeddings.
        Returns empty array with shape (0, embedding_dim) if texts list is empty.

    Note:
        - If ranker model name contains 'e5', prepends "passage: " prefix to each text
          (required by E5 model family)
        - Embeddings are normalized (unit norm)
        - Output dtype is float32
    """
    if not texts:
        return np.zeros((0, ranker.get_sentence_embedding_dimension()), dtype=np.float32)
    enc_inputs = ["passage: " + t if "e5" in RANKER_MODEL_NAME.lower() else t for t in texts]
    emb = ranker.encode(enc_inputs, convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().numpy().astype(np.float32)


def _select_malicious_docs_for_query(
    ranker: SentenceTransformer,
    retrieved_docs: List[str],
    candidate_docs: List[str],
    strategy: str,
    max_docs: int,
) -> List[str]:
    """
    Select malicious documents from candidates using a semantic similarity-based strategy.

    Supports three selection strategies:
    1. "random": Randomly select max_docs candidates
    2. "closest_to_centroid": Select documents closest to the centroid of retrieved docs
    3. "furthest_from_centroid": Select documents furthest from centroid (to maximize diversity)

    Args:
        ranker: SentenceTransformer model for computing embeddings and similarities.
        retrieved_docs: List of documents already retrieved (used as reference for similarity).
        candidate_docs: List of malicious document candidates to select from.
        strategy: Selection strategy ("random", "closest_to_centroid", or "furthest_from_centroid").
        max_docs: Maximum number of documents to select. If None or >= len(candidates),
                 all candidates are returned.

    Returns:
        List of selected malicious documents (up to max_docs items).

    Note:
        - If retrieved_docs is empty, falls back to random selection
        - If candidate_docs is empty, returns empty list
        - Uses embedding centroid of non-empty retrieved documents for similarity computation
        - Handles edge cases gracefully (empty embeddings, no candidates, etc.)
    """
    if not candidate_docs:
        return []

    if max_docs is None or max_docs < 0 or max_docs >= len(candidate_docs):
        target_n = len(candidate_docs)
    else:
        target_n = max_docs

    if target_n == 1:
        if strategy == "random":
            return [random.choice(candidate_docs)]

        nonempty_retrieved = [d for d in retrieved_docs if d and d.strip()]
        if not nonempty_retrieved:
            return [random.choice(candidate_docs)]

        retrieved_emb = encode_texts_with_ranker(ranker, nonempty_retrieved)
        if retrieved_emb.shape[0] == 0:
            return [random.choice(candidate_docs)]

        centroid = retrieved_emb.mean(axis=0, keepdims=True)

        candidate_emb = encode_texts_with_ranker(ranker, candidate_docs)
        if candidate_emb.shape[0] == 0:
            return [random.choice(candidate_docs)]

        sims = np.matmul(candidate_emb, centroid.T).reshape(-1)

        if strategy == "closest_to_centroid":
            idx = int(np.argmax(sims))
        elif strategy == "furthest_from_centroid":
            idx = int(np.argmin(sims))
        else:
            idx = random.randrange(len(candidate_docs))

        return [candidate_docs[idx]]

    if strategy == "random":
        return random.sample(candidate_docs, target_n)

    nonempty_retrieved = [d for d in retrieved_docs if d and d.strip()]
    if not nonempty_retrieved:
        return random.sample(candidate_docs, target_n)

    retrieved_emb = encode_texts_with_ranker(ranker, nonempty_retrieved)
    if retrieved_emb.shape[0] == 0:
        return random.sample(candidate_docs, target_n)

    centroid = retrieved_emb.mean(axis=0, keepdims=True)

    candidate_emb = encode_texts_with_ranker(ranker, candidate_docs)
    if candidate_emb.shape[0] == 0:
        return random.sample(candidate_docs, target_n)

    sims = np.matmul(candidate_emb, centroid.T).reshape(-1)

    if strategy == "closest_to_centroid":
        order = np.argsort(-sims)
    elif strategy == "furthest_from_centroid":
        order = np.argsort(sims)
    else:
        return random.sample(candidate_docs, target_n)

    chosen: List[str] = []
    for idx in order:
        chosen.append(candidate_docs[int(idx)])
        if len(chosen) >= target_n:
            break
    return chosen


def select_malicious_docs_for_batch(
    ranker: SentenceTransformer,
    retrieved_docs_batch_full: List[List[str]],
    malicious_doc_groups_batch: List[List[str]],
    strategy: str,
    max_docs: int,
) -> List[List[str]]:
    """
    Select malicious documents for a batch of queries using semantic similarity strategies.

    Applies malicious document selection independently for each query in the batch.
    Pairs each query's retrieved documents with its malicious document candidates
    and selects the best candidates based on the specified strategy.

    Args:
        ranker: SentenceTransformer model for embedding-based similarity.
        retrieved_docs_batch_full: List of retrieved document lists (one per query).
        malicious_doc_groups_batch: List of malicious document candidate lists (one per query).
        strategy: Selection strategy: "random", "closest_to_centroid", or "furthest_from_centroid".
        max_docs: Maximum number of malicious documents to select per query.

    Returns:
        List of selected malicious document lists (one per query), where each inner list
        contains the selected malicious documents for that query.

    Note:
        - Processes queries independently in the batch
        - Maintains alignment: query i gets docs from retrieved_docs_batch_full[i]
          and candidates from malicious_doc_groups_batch[i]
    """
    chosen_batch: List[List[str]] = []
    for retrieved_docs, candidates in zip(retrieved_docs_batch_full, malicious_doc_groups_batch):
        chosen_docs = _select_malicious_docs_for_query(
            ranker=ranker,
            retrieved_docs=retrieved_docs,
            candidate_docs=candidates,
            strategy=strategy,
            max_docs=max_docs,
        )
        chosen_batch.append(chosen_docs)
    return chosen_batch
