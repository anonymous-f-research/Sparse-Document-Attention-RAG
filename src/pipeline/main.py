from __future__ import annotations

import sys
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from src.pipeline import config
from src.pipeline.utils import prompts
from src.pipeline.attack.content_generation import (
    build_attack_content_for_batch,
    generate_batch_seq2seq,
    load_llm,
)
from src.pipeline.attack.malicious_selection import select_malicious_docs_for_batch
from src.pipeline.attack.doc_corruption import (
    doc_contains_any_gt,
    replace_gt_with_false,
    build_docs_for_attack,
)
from src.pipeline.defenses.cache import load_discern_labels_jsonl, save_discern_labels_jsonl
from src.pipeline.defenses.discern_and_answer import DiscernAndAnswerDefense
from src.pipeline.defenses.none import NoDefense
from src.pipeline.defenses.ragdefender_defense import RagDefenderDefense
from src.pipeline.models.datamodels import PairSpec, QueryData, Resources
from src.pipeline.utils.save_results import save_results
from src.pipeline.utils.metrics import build_pair_metrics, compute_false_answer_stats_for_results
from src.pipeline.retrieval.dense import DenseRetriever, load_faiss_index_and_meta
from src.pipeline.retrieval.sparse import SparseRetriever, load_sparse_searcher
from src.pipeline.retrieval.hybrid import HybridRetriever
from src.pipeline.utils.parsing import load_from_csv
from src.pipeline.utils.ranked_list import attack_config_requests_docs, inject_malicious_docs_into_ranked_list, apply_ranked_list_order
from src.pipeline.sparse_attention_RAG.SDAG import run_rag_with_doc_isolation
from src.pipeline.utils.normalization import exact_match, extract_final_answer


# ---------------------------
# Config Loading
# ---------------------------

def load_json_config(json_path: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a JSON file.

    Args:
        json_path: Path to the JSON configuration file.

    Returns:
        Dictionary of configuration parameters from the JSON file.
        Returns empty dictionary if file doesn't exist or JSON is invalid.

    Note:
        - Invalid JSON or missing files are logged but do not raise exceptions
        - Caller is responsible for validating loaded parameters
    """
    if not json_path or not os.path.exists(json_path):
        print(f"Config JSON not found or path is empty: {json_path}")
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"Loaded config from JSON: {json_path}")
        return cfg
    except Exception as e:
        print(f"Error loading JSON config from {json_path}: {e}")
        return {}


def apply_json_config(json_config: Dict[str, Any]) -> None:
    """
    Apply configuration parameters from a JSON dictionary to the config module.

    Only updates config attributes that:
    1. Exist in the JSON dictionary
    2. Correspond to valid module attributes

    Skips invalid attributes and missing keys silently.

    Args:
        json_config: Dictionary of configuration parameters to apply.

    Returns:
        None. Modifies the config module in-place.

    Note:
        - Type conversion is not performed; JSON types must match config types
        - List parameters (TOP_K, ADD_ATTACK_IN_RANK) are supported
        - Invalid attributes are skipped with a debug message
    """
    if not json_config:
        return

    for key, value in json_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
            print(f"Config: {key} = {value}")
        else:
            print(f"Warning: JSON config key '{key}' does not exist in config module. Skipping.")


# ---------------------------
# Helpers (clean main)
# ---------------------------

def build_pair_specs(top_k_list: Sequence[int], attack_pos_list: Sequence[int]) -> List[PairSpec]:
    """
    Create PairSpec objects by zipping TOP_K and ADD_ATTACK_IN_RANK lists safely.

    Handles mismatched list lengths by using the minimum length and logging a warning.

    Args:
        top_k_list: List of retrieval depths (k values).
        attack_pos_list: List of malicious document insertion positions.

    Returns:
        List of PairSpec objects pairing each top_k with its corresponding attack position.

    Note:
        If lists have different lengths, only the first min(len(top_k_list), len(attack_pos_list))
        pairs are created.
    """
    if len(top_k_list) != len(attack_pos_list):
        pair_count = min(len(top_k_list), len(attack_pos_list))
        print(f"Warning: mismatched list lengths; using first {pair_count} pairs.")
    else:
        pair_count = len(top_k_list)
    return [PairSpec(int(k), int(pos)) for k, pos in zip(top_k_list[:pair_count], attack_pos_list[:pair_count])]


def compute_need_attack_content(preset_false_answer_groups, pairs: Sequence[PairSpec]) -> bool:
    """
    Determine if malicious documents and false answers need to be generated.

    If preset attack content (CSV) is available, generation is skipped.
    Otherwise, generation is needed only if at least one pair requests document injection.

    Args:
        preset_false_answer_groups: Pre-loaded false answer groups from CSV, or None.
        pairs: List of (top_k, attacker_pos) pairs to evaluate.

    Returns:
        True if generation is needed, False if presets are available or no pairs request attacks.
    """
    if preset_false_answer_groups is not None:
        return False
    return any(attack_config_requests_docs(p.attacker_pos) for p in pairs)


def compute_max_k_needed(pairs: Sequence[PairSpec], attack_variant: str) -> int:
    """
    Compute the maximum retrieval depth needed across all pairs.

    For doc_corruption variant, adds 1 to the max top_k to allow document manipulation.

    Args:
        pairs: List of (top_k, attacker_pos) pairs.
        attack_variant: Either 'malicious_doc' or 'doc_corruption'.

    Returns:
        Maximum k needed to retrieve all documents for all pairs.
    """
    max_k = max(p.top_k for p in pairs)
    if attack_variant == "doc_corruption":
        max_k += 1
    return max_k


def load_queries_unified() -> QueryData:
    """
    Load queries and answers from configured dataset source.

    Currently implements CSV loading only.
    Extend this function to add other dataset loaders (NQ, HotpotQA, etc.).

    Returns:
        QueryData object containing queries, answers, and optional attack presets.

    Raises:
        ValueError: If DATASET_NAME is not 'csv' (other loaders not yet implemented).
    """
    if config.DATASET_NAME != "csv":
        raise ValueError("This runner currently implements DATASET_NAME='csv'. Add samplers similarly if needed.")

    questions, short_answers, false_groups, mal_groups, query_ids = load_from_csv(config.CSV_INPUT_PATH)
    num_q = len(questions)
    print(f"Loaded {num_q} queries (grouped by query_id) from CSV: {config.CSV_INPUT_PATH}")

    return QueryData(
        query_ids=query_ids,
        questions=questions,
        short_answers=short_answers,
        false_answer_groups=false_groups,
        malicious_doc_groups=mal_groups,
    )


def init_resources() -> Resources:
    """
    Initialize and load all heavyweight resources (models, indices, searchers).

    Loads:
    - FAISS index and metadata (for dense retrieval)
    - Pyserini BM25 searcher (for sparse retrieval)
    - SentenceTransformer ranker
    - LLM for generation

    Returns:
        Resources object containing all initialized models and indices.

    Note:
        Only loads resources required by the configured RETRIEVER_BACKEND and other settings.
    """
    dense_index = None
    dense_meta = None
    sparse_searcher = None

    if config.RETRIEVER_BACKEND in {"dense", "sparse_and_dense"}:
        print("Loading FAISS index + metadata...")
        dense_index, dense_meta = load_faiss_index_and_meta(config.FAISS_INDEX_PATH, config.META_JSONL_PATH)
        print(f"Index loaded. Metadata entries: {len(dense_meta)}")

    if config.RETRIEVER_BACKEND in {"sparse", "sparse_and_dense"}:
        print("Loading Pyserini BM25 searcher...")
        sparse_searcher = load_sparse_searcher(config.SPARSE_INDEX_NAME_OR_PATH)

    print("Loading ranker...")
    ranker = SentenceTransformer(config.RANKER_MODEL_NAME, device=config.DEVICE)

    print("Loading LLM...")
    tokenizer, llm_model = load_llm(config.LLM_MODEL_NAME, config.DEVICE)

    return Resources(
        ranker=ranker,
        tokenizer=tokenizer,
        llm_model=llm_model,
        dense_index=dense_index,
        dense_meta=dense_meta,
        sparse_searcher=sparse_searcher,
    )


def build_retriever(resources: Resources):
    """
    Factory function to instantiate the appropriate retriever based on RETRIEVER_BACKEND config.

    Args:
        resources: Resources object containing loaded models and indices.

    Returns:
        Retriever instance (DenseRetriever, SparseRetriever, or HybridRetriever).

    Raises:
        ValueError: If RETRIEVER_BACKEND is unknown.
    """
    if config.RETRIEVER_BACKEND == "dense":
        return DenseRetriever(resources.ranker, resources.dense_index, resources.dense_meta)
    if config.RETRIEVER_BACKEND == "sparse":
        return SparseRetriever(resources.sparse_searcher, config.SPARSE_THREADS)
    if config.RETRIEVER_BACKEND == "sparse_and_dense":
        dense = DenseRetriever(resources.ranker, resources.dense_index, resources.dense_meta)
        sparse = SparseRetriever(resources.sparse_searcher, config.SPARSE_THREADS)
        return HybridRetriever(dense, sparse, seed=config.SEED, k0=60)
    raise ValueError(f"Unknown RETRIEVER_BACKEND: {config.RETRIEVER_BACKEND}")


def build_defense():
    """
    Factory function to instantiate the appropriate defense based on DEFENSE_BACKEND config.

    Returns:
        Defense instance (NoDefense, RagDefenderDefense, or DiscernAndAnswerDefense).

    Raises:
        ValueError: If DEFENSE_BACKEND is unknown.
    """
    if config.DEFENSE_BACKEND == "none":
        return NoDefense()
    if config.DEFENSE_BACKEND == "ragdefender":
        return RagDefenderDefense()
    if config.DEFENSE_BACKEND == "discern_and_answer":
        return DiscernAndAnswerDefense()
    raise ValueError(f"Unknown DEFENSE_BACKEND: {config.DEFENSE_BACKEND}")


def num_shuffles_for_prompt_order(order_mode: str) -> int:
    """
    Determine the number of shuffles based on ranking order mode.

    Args:
        order_mode: One of 'top_down', 'bottom_up', or 'random'.

    Returns:
        NUM_RANDOM_SHUFFLES if order_mode is 'random', otherwise 1.
    """
    if order_mode == "random":
        return int(config.NUM_RANDOM_SHUFFLES)
    return 1


# ---------------------------
# NO-ISO generation (finished)
# ---------------------------

def generate_noiso_batch(
    tokenizer,
    llm_model,
    queries: List[str],
    defended_docs_batch: List[List[str]],
    malicious_docs_survived_batch: List[List[str]],
    attacker_pos: int,
    order_mode: str,
) -> List[str]:
    """
    Generate RAG answers without document isolation defense (NO-ISO).

    Builds chat prompts for a batch, injects survived malicious docs at attack position,
    applies ranking reordering policy, and generates answers via batched LLM inference.

    Args:
        tokenizer: Tokenizer with chat_template support.
        llm_model: LLM for answer generation.
        queries: Batch of query strings.
        defended_docs_batch: Retrieved documents after defense filtering.
        malicious_docs_survived_batch: Malicious docs that survived defense (oracle mode).
        attacker_pos: Position to inject malicious docs in the ranked list.
        order_mode: Ranking policy: 'top_down', 'bottom_up', or 'random'.

    Returns:
        List of generated answer strings (one per query).
    """
    rag_prompts: List[str] = []

    for q, docs_ranked, mals in zip(queries, defended_docs_batch, malicious_docs_survived_batch):
        if config.ORACLE:
            ranked_docs = inject_malicious_docs_into_ranked_list(
                base_docs=list(docs_ranked),
                malicious_docs=list(mals),
                attack_pos=attacker_pos,
            )
        else:
            # In non-oracle flow, the ranked list is supposed to already contain the attacker
            # (this runner focuses on ORACLE=True as in your current snippet usage).
            ranked_docs = list(docs_ranked)

        ranked_docs = apply_ranked_list_order(ranked_docs, order_mode)

        docs_text = "\n\n".join(f"- {d.strip()}" for d in ranked_docs if d and d.strip())
        user_content = prompts.USER_RAG_PROMPT.format(query=q, docs_text=docs_text)

        chat_str = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompts.SYSTEM_PROMPT_RAG},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        rag_prompts.append(chat_str)

    # batched generation
    answers: List[str] = []
    for j in range(0, len(rag_prompts), config.LLM_BATCH_SIZE):
        sub = rag_prompts[j:j + config.LLM_BATCH_SIZE]
        out = generate_batch_seq2seq(
            tokenizer,
            llm_model,
            sub,
            max_tokens=config.MAX_GEN_TOKENS_RAG,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
        )
        answers.extend(out)

    return answers


# ---------------------------
# Pair execution (malicious_doc)
# ---------------------------

def run_pair_malicious_doc_attack_for_batch(
    *,
    resources: Resources,
    defense,
    discern_cache: Dict[Tuple[str, str], str],
    pair: PairSpec,
    batch_qids: List[str],
    batch_qs: List[str],
    batch_gts: List[List[str]],
    retrieval_docs_full: List[List[str]],
    retrieval_ids_full: List[List[str]],
    retrieval_scores_full: List[List[float]],
    false_groups_batch: List[List[str]],
    chosen_mals_batch: List[List[str]],
) -> List[Dict[str, Any]]:
    """
    Execute malicious document injection attack for a batch of queries.

    Process:
    1. Extract corpus pool of top-(k+1) documents
    2. Apply defense to filter adversarial documents
    3. Generate ISO and NO-ISO answers
    4. Evaluate correctness against ground truth and false answers

    Args:
        resources: Loaded models and indices.
        defense: Defense instance to filter documents.
        discern_cache: Cache for Discern defense classifications.
        pair: (top_k, attacker_pos) specification.
        batch_qids: Query identifiers for this batch.
        batch_qs: Query texts.
        batch_gts: Ground truth answers (lists per query).
        retrieval_docs_full: Retrieved documents from full ranking.
        retrieval_ids_full: Retrieved document IDs.
        retrieval_scores_full: Retrieved document relevance scores.
        false_groups_batch: False answers per query.
        chosen_mals_batch: Selected malicious docs per query.

    Returns:
        List of result dictionaries (one per query), each containing:
            - query_id, question, short_answers
            - false_answer, malicious_doc
            - rag_answer_iso, rag_answer_noiso
            - ground_truth_match_iso/noiso, false_match_iso/noiso
    """
    k_plus_one = pair.top_k + 1

    # Build corpus pool = first (k+1)
    corpus_docs_pool_batch = [docs[:k_plus_one] for docs in retrieval_docs_full]
    corpus_ids_pool_batch = [ids_[:k_plus_one] for ids_ in retrieval_ids_full]
    corpus_scores_pool_batch = [sc[:k_plus_one] for sc in retrieval_scores_full]

    # Apply defense (oracle: pass malicious docs; non-oracle: pass [])
    defended_docs_batch: List[List[str]] = []
    defended_ids_batch: List[List[str]] = []
    defended_scores_batch: List[List[float]] = []
    survived_mals_batch: List[List[str]] = []
    discern_labels_batch: List[Optional[Dict[str, str]]] = []

    for qid, q, docs_pool, ids_pool, scores_pool, mals in zip(
        batch_qids, batch_qs, corpus_docs_pool_batch, corpus_ids_pool_batch, corpus_scores_pool_batch, chosen_mals_batch
    ):
        out = defense.apply(
            query_id=str(qid),
            query=q,
            corpus_docs=list(docs_pool),
            corpus_ids=list(ids_pool),
            corpus_scores=list(scores_pool),
            malicious_docs=list(mals) if config.ORACLE else [],
            does_oracle=config.ORACLE,
            persistent_cache=discern_cache,
        )
        defended_docs_batch.append(list(out.ranked_docs))
        defended_ids_batch.append(list(out.ranked_ids))
        defended_scores_batch.append(list(out.ranked_scores) if out.ranked_scores is not None else list(scores_pool))
        survived_mals_batch.append(list(out.malicious_docs_survived))
        discern_labels_batch.append(out.doc_labels)

    shuffles = num_shuffles_for_prompt_order(config.RNAKED_LIST_ORDER_IN_PROMPT)
    results_rows: List[Dict[str, Any]] = []

    for _shuffle_idx in range(shuffles):
        # ISO generation: per query (ISO uses custom mask; easiest to run per item)
        answers_iso: List[str] = []
        for q, docs_ranked, mals_survived in zip(batch_qs, defended_docs_batch, survived_mals_batch):
            if config.ORACLE:
                ans = run_rag_with_doc_isolation(
                    model=resources.llm_model,
                    tokenizer=resources.tokenizer,
                    query=q,
                    malicious_docs=mals_survived,
                    retrieved_docs=docs_ranked,
                    max_new_tokens=config.MAX_GEN_TOKENS_RAG,
                    device=config.DEVICE,
                    add_attack_in_rank=pair.attacker_pos,
                    ranker=resources.ranker,
                    doc_neighbors_k=config.DOC_NEIGHBORS_K,
                )
            else:
                ans = run_rag_with_doc_isolation(
                    model=resources.llm_model,
                    tokenizer=resources.tokenizer,
                    query=q,
                    malicious_docs=[],
                    retrieved_docs=docs_ranked,
                    max_new_tokens=config.MAX_GEN_TOKENS_RAG,
                    device=config.DEVICE,
                    add_attack_in_rank=0,
                    ranker=resources.ranker,
                    doc_neighbors_k=config.DOC_NEIGHBORS_K,
                )
            answers_iso.append(ans)

        # NO-ISO generation: batched
        answers_noiso = generate_noiso_batch(
            tokenizer=resources.tokenizer,
            llm_model=resources.llm_model,
            queries=batch_qs,
            defended_docs_batch=defended_docs_batch,
            malicious_docs_survived_batch=survived_mals_batch,
            attacker_pos=pair.attacker_pos,
            order_mode=config.RNAKED_LIST_ORDER_IN_PROMPT,
        )

        # Collect rows
        for qid, q, gts, fa_list, docs_ranked, ids_ranked, mals_survived, ans_iso, ans_noiso in zip(
            batch_qids,
            batch_qs,
            batch_gts,
            false_groups_batch,
            defended_docs_batch,
            defended_ids_batch,
            survived_mals_batch,
            answers_iso,
            answers_noiso,
        ):
            ans_iso_clean = extract_final_answer(ans_iso)
            ans_noiso_clean = extract_final_answer(ans_noiso)

            gt_match_iso = any(exact_match(ans_iso_clean, gt) for gt in gts)
            gt_match_noiso = any(exact_match(ans_noiso_clean, gt) for gt in gts)

            false_match_iso = any(exact_match(ans_iso_clean, fa) for fa in fa_list) if fa_list else False
            false_match_noiso = any(exact_match(ans_noiso_clean, fa) for fa in fa_list) if fa_list else False

            # for logging: join malicious docs that survived defense (oracle)
            has_attack = attack_config_requests_docs(pair.attacker_pos)
            mal_str = " ||| ".join(mals_survived) if (has_attack and mals_survived) else ""

            results_rows.append({
                "query_id": qid,
                "question": q,
                "short_answers": gts,
                "false_answer": fa_list,
                "malicious_doc": mal_str,
                "retrieved_docs": list(docs_ranked),
                "retrieved_doc_ids": list(ids_ranked),
                "rag_answer_iso": ans_iso_clean,
                "rag_answer_noiso": ans_noiso_clean,
                "ground_truth_match_iso": gt_match_iso,
                "ground_truth_match_noiso": gt_match_noiso,
                "false_match_iso": false_match_iso,
                "false_match_noiso": false_match_noiso,
            })

    return results_rows


# ---------------------------
# Pair execution (doc_corruption)
# ---------------------------

def run_pair_doc_corruption_for_batch(
    *,
    resources: Resources,
    pair: PairSpec,
    batch_qids: List[str],
    batch_qs: List[str],
    batch_gts: List[List[str]],
    retrieval_docs_full: List[List[str]],
    retrieval_ids_full: List[List[str]],
    false_groups_batch: List[List[str]],
) -> List[Dict[str, Any]]:
    """
    Execute document corruption attack for a batch of queries.

    Process:
    1. For each query, find documents containing ground truth answers
    2. Replace ground truth with false answer
    3. Reorder document to attack position
    4. Generate ISO and NO-ISO answers
    5. Log results (one row per attacked document)

    Args:
        resources: Loaded models and indices.
        pair: (top_k, attacker_pos) specification.
        batch_qids: Query identifiers.
        batch_qs: Query texts.
        batch_gts: Ground truth answers (lists per query).
        retrieval_docs_full: Retrieved documents from full ranking.
        retrieval_ids_full: Retrieved document IDs.
        false_groups_batch: False answers per query.

    Returns:
        List of result dictionaries (one per attacked document per query).
    """
    k_plus_one = pair.top_k + 1
    shuffles = num_shuffles_for_prompt_order(config.RNAKED_LIST_ORDER_IN_PROMPT)

    results_rows: List[Dict[str, Any]] = []

    for _shuffle_idx in range(shuffles):
        for qid, q, gts, docs_full, ids_full, fa_list in zip(
            batch_qids, batch_qs, batch_gts, retrieval_docs_full, retrieval_ids_full, false_groups_batch
        ):
            if not fa_list:
                # In your snippet you assumed CSV provides it
                continue
            false_ans = fa_list[0]

            docs_pool = docs_full[:k_plus_one]
            ids_pool = ids_full[:k_plus_one]

            candidate_indices = [j for j, d in enumerate(docs_pool) if d and doc_contains_any_gt(d, gts)]
            if not candidate_indices:
                continue

            for attacked_idx in candidate_indices:
                original_doc = docs_pool[attacked_idx]
                poisoned_doc = replace_gt_with_false(original_doc, gts, false_ans)

                docs_pool_poisoned = list(docs_pool)
                docs_pool_poisoned[attacked_idx] = poisoned_doc

                docs_for_prompt = build_docs_for_attack(
                    docs=docs_pool_poisoned,
                    attacked_idx=attacked_idx,
                    attack_pos=pair.attacker_pos,
                    top_k=k_plus_one,
                )
                ids_for_prompt = build_docs_for_attack(
                    docs=ids_pool,
                    attacked_idx=attacked_idx,
                    attack_pos=pair.attacker_pos,
                    top_k=k_plus_one,
                )

                # ISO: since docs already reordered, we do not insert again
                ans_iso = run_rag_with_doc_isolation(
                    model=resources.llm_model,
                    tokenizer=resources.tokenizer,
                    query=q,
                    malicious_docs=[],
                    retrieved_docs=docs_for_prompt,
                    max_new_tokens=config.MAX_GEN_TOKENS_RAG,
                    device=config.DEVICE,
                    add_attack_in_rank=0,
                    ranker=resources.ranker,
                    doc_neighbors_k=config.DOC_NEIGHBORS_K,
                )

                # NO-ISO: build standard prompt from docs_for_prompt
                ranked_docs = apply_ranked_list_order(list(docs_for_prompt), config.RNAKED_LIST_ORDER_IN_PROMPT)
                docs_text = "\n\n".join(f"- {d.strip()}" for d in ranked_docs if d and d.strip())
                user_content = prompts.USER_RAG_PROMPT.format(query=q, docs_text=docs_text)

                chat_str = resources.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": prompts.SYSTEM_PROMPT_RAG},
                        {"role": "user", "content": user_content},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                out = generate_batch_seq2seq(
                    resources.tokenizer,
                    resources.llm_model,
                    [chat_str],
                    max_tokens=config.MAX_GEN_TOKENS_RAG,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                )
                ans_noiso = out[0] if out else ""

                ans_iso_clean = extract_final_answer(ans_iso)
                ans_noiso_clean = extract_final_answer(ans_noiso)

                gt_match_iso = any(exact_match(ans_iso_clean, gt) for gt in gts)
                gt_match_noiso = any(exact_match(ans_noiso_clean, gt) for gt in gts)

                false_match_iso = exact_match(ans_iso_clean, false_ans)
                false_match_noiso = exact_match(ans_noiso_clean, false_ans)

                # store poisoned doc for inspection
                results_rows.append({
                    "query_id": qid,
                    "question": q,
                    "short_answers": gts,
                    "false_answer": [false_ans],
                    "malicious_doc": poisoned_doc,
                    "retrieved_docs": [d for i, d in enumerate(docs_pool) if i != attacked_idx],
                    "retrieved_doc_ids": list(ids_for_prompt),
                    "rag_answer_iso": ans_iso_clean,
                    "rag_answer_noiso": ans_noiso_clean,
                    "ground_truth_match_iso": gt_match_iso,
                    "ground_truth_match_noiso": gt_match_noiso,
                    "false_match_iso": false_match_iso,
                    "false_match_noiso": false_match_noiso,
                })

    return results_rows


# ---------------------------
# MAIN (clean)
# ---------------------------

def main(config_json_path: Optional[str] = None) -> None:
    """
    Main entry point for the RAG attack pipeline.

    Executes the full attack workflow:
    1. Load and apply JSON config if provided
    2. Initialize random seeds for reproducibility
    3. Load queries and dataset
    4. Initialize models and retrievers
    5. Generate attack content if needed
    6. For each (top_k, attacker_pos) pair:
       - Retrieve documents
       - Select malicious documents
       - Run malicious_doc or doc_corruption attacks
       - Evaluate and save results

    Args:
        config_json_path: Optional path to JSON config file. If provided, overrides hardcoded config.
                         Falls back to config module defaults for missing parameters.

    Returns:
        None. Results are saved to CSV and JSON files.

    Side Effects:
        - Initializes random seeds globally
        - Creates and writes output CSV and JSON files
        - Prints progress and metrics to stdout
    """
    # Load and apply JSON config if provided
    if config_json_path:
        json_cfg = load_json_config(config_json_path)
        apply_json_config(json_cfg)

    config.init_seeds()
    config.validate_config()

    query_data = load_queries_unified()
    pairs = build_pair_specs(config.TOP_K, config.ADD_ATTACK_IN_RANK)
    if not pairs:
        print("No pairs to run. Exiting.")
        return

    num_q = len(query_data.questions)
    if num_q == 0:
        print("No queries. Exiting.")
        return

    resources = init_resources()
    retriever = build_retriever(resources)
    defense = build_defense()

    discern_cache: Dict[Tuple[str, str], str] = {}
    if config.DISCERN_LABELS_LOAD_PATH:
        discern_cache = load_discern_labels_jsonl(config.DISCERN_LABELS_LOAD_PATH)

    need_attack_content = compute_need_attack_content(query_data.false_answer_groups, pairs)
    max_k_needed = compute_max_k_needed(pairs, config.ATTACK_VARIANT)

    # One bucket per pair
    results_per_pair: Dict[Tuple[int, int], List[Dict[str, Any]]] = {(p.top_k, p.attacker_pos): [] for p in pairs}

    # Batch over queries (retrieve once with max_k_needed)
    for i in tqdm(range(0, num_q, config.BATCH_SIZE_EMBED_Q), desc="Processing query batches"):
        batch_qs = query_data.questions[i:i + config.BATCH_SIZE_EMBED_Q]
        batch_gts = query_data.short_answers[i:i + config.BATCH_SIZE_EMBED_Q]
        batch_qids = query_data.query_ids[i:i + config.BATCH_SIZE_EMBED_Q]

        # Retrieve once at max_k_needed
        retrieval = retriever.retrieve_batch(batch_qs, max_k_needed=max_k_needed, embed_batch_size=config.BATCH_SIZE_EMBED_Q)
        retrieval_docs_full = retrieval.docs_texts_full
        retrieval_ids_full = retrieval.ids_full
        retrieval_scores_full = retrieval.scores_full

        # Prepare false answers / malicious docs for this batch
        if query_data.false_answer_groups is not None and query_data.malicious_doc_groups is not None:
            false_groups_batch = query_data.false_answer_groups[i:i + config.BATCH_SIZE_EMBED_Q]
            mal_groups_batch = query_data.malicious_doc_groups[i:i + config.BATCH_SIZE_EMBED_Q]
        else:
            false_groups_batch, mal_groups_batch = build_attack_content_for_batch(
                preset_false_answer_groups=None,
                preset_malicious_doc_groups=None,
                need_attack_content=need_attack_content,
                tokenizer=resources.tokenizer,
                model=resources.llm_model,
                queries=batch_qs,
            )

        # Choose malicious docs (possibly multiple) per query based on strategy
        chosen_mals_batch = select_malicious_docs_for_batch(
            ranker=resources.ranker,
            retrieved_docs_batch_full=retrieval_docs_full,
            malicious_doc_groups_batch=mal_groups_batch,
            strategy=config.MALICIOUS_DOC_SELECTION_STRATEGY,
            max_docs=config.MAX_MALICIOUS_DOCS_PER_QUERY,
        )

        # Run each pair on this batch
        for pair in pairs:
            key = (pair.top_k, pair.attacker_pos)

            if config.ATTACK_VARIANT == "malicious_doc":
                rows = run_pair_malicious_doc_attack_for_batch(
                    resources=resources,
                    defense=defense,
                    discern_cache=discern_cache,
                    pair=pair,
                    batch_qids=batch_qids,
                    batch_qs=batch_qs,
                    batch_gts=batch_gts,
                    retrieval_docs_full=retrieval_docs_full,
                    retrieval_ids_full=retrieval_ids_full,
                    retrieval_scores_full=retrieval_scores_full,
                    false_groups_batch=false_groups_batch,
                    chosen_mals_batch=chosen_mals_batch,
                )
                results_per_pair[key].extend(rows)

            elif config.ATTACK_VARIANT == "doc_corruption":
                rows = run_pair_doc_corruption_for_batch(
                    resources=resources,
                    pair=pair,
                    batch_qids=batch_qids,
                    batch_qs=batch_qs,
                    batch_gts=batch_gts,
                    retrieval_docs_full=retrieval_docs_full,
                    retrieval_ids_full=retrieval_ids_full,
                    false_groups_batch=false_groups_batch,
                )
                results_per_pair[key].extend(rows)

            else:
                raise ValueError(f"Unknown ATTACK_VARIANT: {config.ATTACK_VARIANT}")

    # Save one CSV + one JSON per pair
    for pair in pairs:
        key = (pair.top_k, pair.attacker_pos)
        results = results_per_pair[key]

        out_csv = f"{config.OUTPUT_CSV_BASE}_top_k={pair.top_k}_attacker_pos={pair.attacker_pos}.csv"
        save_results(results, out_csv)
        print(f"Saved: {out_csv}")

        metrics = build_pair_metrics(results, pair.top_k, pair.attacker_pos)
        metrics["false_answer_stats"] = compute_false_answer_stats_for_results(results)

        # Attach config snapshot
        metrics["run_config"] = config.export_config_dict()

        out_json = f"{config.OUTPUT_CSV_BASE}_top_k={pair.top_k}_attacker_pos={pair.attacker_pos}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON: {out_json}")

    # Persist discern cache if used
    if config.DEFENSE_BACKEND == "discern_and_answer":
        out_labels_path = f"{config.OUTPUT_CSV_BASE}_{config.DISCERN_LABELS_SAVE_SUFFIX}"
        save_discern_labels_jsonl(out_labels_path, discern_cache)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_json_path=config_path)

