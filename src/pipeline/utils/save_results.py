from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

def save_results(results: List[Dict[str, Any]], csv_path: str):
    """
    Evaluate RAG attack results and save them to a CSV file with performance metrics.

    This function processes a list of attack result dictionaries, writes them to a CSV file,
    and prints summary statistics including accuracy and attack success rates for both
    isolated (ISO) and non-isolated (NO-ISO) answer generation modes.

    Args:
        results: List of result dictionaries, each containing:
                 - query_id: Unique identifier for the query
                 - question: The query text
                 - short_answers: List of ground-truth answer strings
                 - false_answer: The false answer used in the attack (optional)
                 - malicious_doc: The injected malicious document (optional)
                 - retrieved_docs: List of retrieved document texts
                 - retrieved_doc_ids: List of retrieved document IDs
                 - rag_answer_iso: RAG-generated answer with doc isolation defense
                 - rag_answer_noiso: RAG-generated answer without doc isolation
                 - ground_truth_match_iso: Boolean indicating if true answer appears in ISO output
                 - ground_truth_match_noiso: Boolean indicating if true answer appears in NO-ISO output
                 - false_match_iso: Boolean indicating if false answer appears in ISO output
                 - false_match_noiso: Boolean indicating if false answer appears in NO-ISO output
        csv_path: File path where the CSV results will be written.

    Returns:
        None. Outputs are written to csv_path and printed to stdout.

    Side Effects:
        - Creates/overwrites CSV file at csv_path
        - Prints summary statistics to stdout:
          * Total number of queries
          * Ground truth matching accuracy for ISO and NO-ISO modes
          * Attack success rate (false answer matching) for both modes
    """
    fieldnames = [
        "query_id",
        "question",
        "short_answers",
        "false_answer",
        "malicious_doc",
        "retrieved_docs",
        "retrieved_doc_ids",
        "rag_answer_iso",
        "rag_answer_noiso",
        "ground_truth_match_iso",
        "ground_truth_match_noiso",
        "false_match_iso",
        "false_match_noiso",
    ]

    # Create output directory if needed
    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(csv_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "query_id": r.get("query_id", ""),
                "question": r["question"],
                "short_answers": "\n".join(r["short_answers"]),
                "false_answer": r.get("false_answer", ""),
                "malicious_doc": r.get("malicious_doc", ""),
                "retrieved_docs": " ||| ".join(r["retrieved_docs"]),
                "retrieved_doc_ids": "\n".join(r["retrieved_doc_ids"]),
                "rag_answer_iso": r["rag_answer_iso"],
                "rag_answer_noiso": r["rag_answer_noiso"],
                "ground_truth_match_iso": int(r["ground_truth_match_iso"]),
                "ground_truth_match_noiso": int(r["ground_truth_match_noiso"]),
                "false_match_iso": int(r["false_match_iso"]),
                "false_match_noiso": int(r["false_match_noiso"]),
            })

    total = len(results)
    gt_correct_iso = sum(1 for r in results if r["ground_truth_match_iso"])
    gt_correct_noiso = sum(1 for r in results if r["ground_truth_match_noiso"])
    attack_success_iso = sum(1 for r in results if r["false_match_iso"])
    attack_success_noiso = sum(1 for r in results if r["false_match_noiso"])

    print(f"Total queries: {total}")
    print(f"[ISO]    true answer rate: {gt_correct_iso}/{total} = {gt_correct_iso/total:.3f}")
    print(f"[NO-ISO] true answer rate: {gt_correct_noiso}/{total} = {gt_correct_noiso/total:.3f}")
    print(f"[ISO]    attack success rate: {attack_success_iso}/{total} = {attack_success_iso/total:.3f}")
    print(f"[NO-ISO] attack success rate: {attack_success_noiso}/{total} = {attack_success_noiso/total:.3f}")
