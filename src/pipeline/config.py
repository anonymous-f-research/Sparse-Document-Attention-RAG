"""
Central configuration for the attack_pipeline.

Keep all constants here so your modules stay clean and testable.
This file keeps your current logic/config values exactly as-is.
"""

from __future__ import annotations
import multiprocessing
import random
from typing import Optional, Any, Dict
import numpy as np
import torch
from transformers import set_seed

# =========================
# CONFIG (copied from your script)
# =========================
# Random seed for reproducibility across all libraries (numpy, torch, random, transformers).
SEED = 42

# Number of queries to sample from the dataset for evaluation.
SAMPLE_SIZE = 100

# List of retrieval depths to evaluate (number of documents to retrieve from the corpus).
TOP_K = [5]
# List of positions where malicious documents are injected into the ranked list.
# Positive values: fixed rank position (1-indexed).
# -1: random positions.
# 0: no injection.
ADD_ATTACK_IN_RANK = [1]
# Batch size for embedding queries during dense retrieval.
BATCH_SIZE_EMBED_Q = 32
# Batch size for LLM text generation tasks.
LLM_BATCH_SIZE = 4
# File path to the FAISS index file for dense retrieval backend.
FAISS_INDEX_PATH = "faiss.index"
# File path to the metadata JSONL file containing document metadata indexed by FAISS.
META_JSONL_PATH = "docs_meta.jsonl"
# HuggingFace model identifier for the sentence transformer ranker (e.g., E5, BGE).
RANKER_MODEL_NAME = "intfloat/e5-large-v2"
# HuggingFace model identifier for the LLM used in RAG and content generation tasks.
LLM_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# Device for model inference: 'cuda' if GPU available, else 'cpu'.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Dataset split to use: 'train', 'validation', or 'test' (dataset-specific).
DATASET_SPLIT = "validation"
# Base name for output CSV files (actual filename includes top_k and attacker_pos).
OUTPUT_CSV_BASE = "attack_results"
# Path to JSON file containing pre-sampled query IDs and questions.
SAMPLED_QUERIES_JSON = "sampled_nq_queries.json"
# Maximum tokens to generate for false answer content generation.
MAX_GEN_TOKENS_false_answer = 50
# Maximum tokens to generate for malicious document content generation.
MAX_GEN_TOKENS_document = 250
# Maximum tokens to generate for RAG answer generation.
MAX_GEN_TOKENS_RAG = 500
# Temperature parameter for LLM generation (lower = more deterministic, higher = more random).
TEMPERATURE = 0.1
# Top-p (nucleus sampling) parameter for LLM generation (controls diversity via cumulative probability).
TOP_P = 1.0

# Dataset source: 'csv' (local CSV file), 'nq' (Natural Questions), 'hotpotqa', etc.
DATASET_NAME = "csv"
# Attack variant: 'malicious_doc' (inject documents) or 'doc_corruption' (poison existing documents).
ATTACK_VARIANT = "malicious_doc"
# Path to input CSV file containing queries, answers, and optional false answers/malicious docs.
CSV_INPUT_PATH = "input.csv"
# Order mode for documents in RAG prompt: 'top_down' (original), 'bottom_up' (reversed), or 'random' (shuffled).
RNAKED_LIST_ORDER_IN_PROMPT = "top_down"
# Number of random shuffles to apply when RNAKED_LIST_ORDER_IN_PROMPT='random' for statistical robustness.
NUM_RANDOM_SHUFFLES = 10
# Number of document neighbors to include in document isolation attention masking (0 = strict isolation).
DOC_NEIGHBORS_K = 0
# Strategy for selecting malicious documents from candidates: 'random', 'closest_to_centroid', or 'furthest_from_centroid'.
MALICIOUS_DOC_SELECTION_STRATEGY = "random"
# Maximum number of malicious documents to inject per query.
MAX_MALICIOUS_DOCS_PER_QUERY = 1
# Oracle mode: if True, defense sees malicious docs and can filter them. If False, defense sees only corpus docs.
ORACLE = True

# =========================
# Retrieval backend
# =========================

# Retrieval system: 'dense', 'sparse', or 'sparse_and_dense' (hybrid with RRF).
RETRIEVER_BACKEND = "dense"
# Pyserini index name or local path for sparse retrieval (BM25) backend.
SPARSE_INDEX_NAME_OR_PATH = "wikipedia-dpr-100w"
# Number of threads for parallel BM25 batch search (None = CPU count).
SPARSE_THREADS: Optional[int] = None

# =========================
# Defense configuration
# =========================

# Defense mechanism: 'none' (no defense), 'ragdefender' (RAGDefender model), or 'discern_and_answer' (OpenAI classifier).
DEFENSE_BACKEND = "none"
# Task type for RAGDefender: 'single_hop' or 'multi_hop' depending on question complexity.
RAGDEFENDER_TASK = ""
# Device for RAGDefender model: 'cuda' or 'cpu'.
RAGDEFENDER_DEVICE = DEVICE
# OpenAI model name for Discern classifier (e.g., 'gpt-3.5-turbo', 'gpt-4').
DISCERN_CLASSIFIER_MODEL = ""
# OpenAI API key for Discern defense (leave empty to disable).
DISCERN_OPENAI_API_KEY = ""
# Maximum number of documents to classify per query (for efficiency with expensive API calls).
DISCERN_MAX_DOCS_TO_CLASSIFY = 32
# Temperature for Discern classifier generation (0.0 = deterministic).
DISCERN_CLASSIFY_TEMPERATURE = 0.0
# Path to pre-loaded Discern classification labels (JSONL format), or empty to start fresh.
DISCERN_LABELS_LOAD_PATH = ""
# Suffix for output file containing saved Discern classification labels.
DISCERN_LABELS_SAVE_SUFFIX = ""


def init_seeds() -> None:
    """Reproduce runs."""
    global SPARSE_THREADS
    if SPARSE_THREADS is None:
        SPARSE_THREADS = multiprocessing.cpu_count()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_seed(SEED)


def validate_config() -> None:
    """Keep your original config checks."""
    if RETRIEVER_BACKEND in {"sparse_and_dense"} and not ORACLE:
        raise ValueError("Sparse retrieval currently supports ORACLE=True only (NON_ORACLE is disabled).")


def export_config_dict() -> Dict[str, Any]:
    """
    Export a JSON-serializable snapshot of the current config module.
    Filters out private names, modules, and callables.
    """
    out: Dict[str, Any] = {}
    for k, v in globals().items():
        if k.startswith("_"):
            continue
        if callable(v):
            continue
        if k in {"export_config_dict"}:
            continue

        # best-effort JSON-serializable filtering
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = v
        else:
            out[k] = str(v)
    return out
