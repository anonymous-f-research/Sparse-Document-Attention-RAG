from __future__ import annotations
import json
from typing import List, Sequence, Tuple

from pyserini.search.lucene import LuceneSearcher

from src.pipeline.models.datamodels import RetrievalBatch
from src.pipeline.retrieval.retriever import Retriever


def load_sparse_searcher(index_name_or_path: str) -> LuceneSearcher:
    """
    Load a Lucene-based BM25 searcher from either a prebuilt Pyserini index or a local directory.

    This function attempts to load a prebuilt Pyserini index first. If that fails,
    it tries to load from a local index directory path.

    Args:
        index_name_or_path: Either a prebuilt index name (e.g., 'msmarco-v1-passage')
                           or a path to a local Lucene index directory.

    Returns:
        A LuceneSearcher instance ready for BM25 search queries.

    Raises:
        ValueError: If neither a prebuilt index nor a local index can be found.
    """
    try:
        print("Attempting to load Pyserini prebuilt index...")
        return LuceneSearcher.from_prebuilt_index(index_name_or_path)
    except ValueError:
        print("Not a prebuilt index. Attempting to load as a local index directory...")
        return LuceneSearcher(index_name_or_path)


def bm25_batch_search(
    searcher: LuceneSearcher,
    queries: List[str],
    k: int,
    threads: int,
) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
    """
    Perform BM25 batch search on multiple queries using Pyserini's LuceneSearcher.

    Args:
        searcher: Initialized LuceneSearcher instance for BM25 search.
        queries: List of query strings to search for.
        k: Number of top results to retrieve per query.
        threads: Number of threads to use for parallel batch searching.

    Returns:
        A tuple containing:
            - retrieved_texts_batch: List of lists containing document texts (one list per query).
            - retrieved_ids_batch: List of lists containing document IDs (one list per query).
            - retrieved_scores_batch: List of lists containing BM25 scores (one list per query).

    Note:
        - If a query retrieves fewer than k results, the lists are padded with empty strings,
          "NA" IDs, and -inf scores to maintain consistent batch dimensions.
        - Documents are extracted from hit.raw (precompiled JSON) or fetched via searcher.doc().
        - The raw field is expected to be JSON with "contents" or "text" keys.
    """
    qids = [str(i) for i in range(len(queries))]
    hits_by_qid = searcher.batch_search(queries, qids=qids, k=k, threads=threads)

    retrieved_texts_batch: List[List[str]] = []
    retrieved_ids_batch: List[List[str]] = []
    retrieved_scores_batch: List[List[float]] = []

    for i in range(len(queries)):
        qid = str(i)
        hits = hits_by_qid.get(qid, [])

        texts: List[str] = []
        ids_: List[str] = []
        scores: List[float] = []

        for hit in hits:
            docid = hit.docid
            score = float(hit.score)

            raw = hit.raw
            if raw is None:
                doc = searcher.doc(docid)
                raw = doc.raw() if doc is not None else None

            text = ""
            if raw:
                try:
                    obj = json.loads(raw)
                    text = obj.get("contents", "") or obj.get("text", "") or ""
                except Exception:
                    text = raw

            texts.append(text)
            ids_.append(str(docid))
            scores.append(score)

        while len(texts) < k:
            texts.append("")
            ids_.append("NA")
            scores.append(float("-inf"))

        retrieved_texts_batch.append(texts[:k])
        retrieved_ids_batch.append(ids_[:k])
        retrieved_scores_batch.append(scores[:k])

    return retrieved_texts_batch, retrieved_ids_batch, retrieved_scores_batch


class SparseRetriever(Retriever):
    """
    Sparse retrieval system using BM25 via Pyserini's Lucene backend.

    This retriever performs lexical matching-based search (BM25) on an indexed corpus.
    It is efficient for large-scale document collections but may miss semantically
    similar documents that use different vocabulary.

    Attributes:
        searcher: LuceneSearcher instance for performing BM25 searches.
        threads: Number of threads for parallel batch search operations.
    """

    def __init__(self, searcher: LuceneSearcher, threads: int) -> None:
        """
        Initialize the SparseRetriever.

        Args:
            searcher: Initialized LuceneSearcher for BM25 search operations.
            threads: Number of threads to use for batch search (higher values speed up batch processing).
        """
        self.searcher = searcher
        self.threads = threads

    def retrieve_batch(self, queries: Sequence[str], max_k_needed: int, embed_batch_size: int) -> RetrievalBatch:
        """
        Retrieve top-k documents for a batch of queries using BM25.

        Args:
            queries: Sequence of query strings to search for.
            max_k_needed: Maximum number of documents to retrieve per query.
            embed_batch_size: Batch size for embedding (unused in sparse retrieval, kept for interface compatibility).

        Returns:
            A RetrievalBatch object containing:
                - q_embs: None for sparse retrieval (no embeddings are computed).
                - docs_texts_full: Retrieved document texts for each query.
                - ids_full: Document IDs for retrieved documents.
                - scores_full: BM25 scores for retrieved documents.
        """
        # q_embs are unused in sparse path
        q_embs = [None] * len(queries)
        docs_texts, doc_ids, doc_scores = bm25_batch_search(
            searcher=self.searcher,
            queries=list(queries),
            k=max_k_needed,
            threads=self.threads,
        )
        return RetrievalBatch(q_embs=q_embs, docs_texts_full=docs_texts, ids_full=doc_ids, scores_full=doc_scores)
