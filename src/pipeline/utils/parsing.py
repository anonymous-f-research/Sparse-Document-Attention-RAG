from __future__ import annotations

import ast
import csv
import json
from typing import Any, Dict, List, Tuple


def parse_list_field(x: str) -> List[str]:
    """
    Parse a list from a CSV cell.
    Accepts JSON (["a","b"]) or Python-ish lists (['a','b']).
    Falls back to splitting by '|||' or commas.
    """
    if x is None:
        return []
    x = x.strip()
    if not x:
        return []
    try:
        val = json.loads(x)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    try:
        val = ast.literal_eval(x)
        if isinstance(val, list):
            return [str(v) for v in val]
    except Exception:
        pass
    if "|||" in x:
        return [t.strip() for t in x.split("|||") if t.strip()]
    if "," in x:
        return [t.strip() for t in x.split(",") if t.strip()]
    return [x]


def load_from_csv(
    csv_path: str,
    match_field_for_groups: str = "query",
) -> Tuple[List[str], List[List[str]], List[List[str]], List[List[str]], List[str]]:
    """
    Returns (grouped by match_field_for_groups):
      queries:               List[str]
      gt_answers_list:       List[List[str]]
      false_answers_groups:  List[List[str]]
      malicious_docs_groups: List[List[str]]
      query_ids:             List[str]

    Required columns:
      query, query_id, ground_truth_answers, false_answer, malicious_document
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"query", "query_id", "ground_truth_answers", "false_answer", "malicious_document"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        grouped: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        for row in reader:
            qid = str(row["query_id"]).strip()
            query = (row["query"] or "").strip()
            gt_list = parse_list_field(row["ground_truth_answers"])
            false_ans = (row.get("false_answer") or "").strip()
            mal_doc = (row.get("malicious_document") or "").strip()

            match_key = row[match_field_for_groups]
            if match_key not in grouped:
                grouped[match_key] = {
                    "query_id": qid,
                    "query": query,
                    "ground_truth_answers": gt_list,
                    "false_answers": [],
                    "malicious_docs": [],
                }
                order.append(match_key)

            if false_ans and false_ans not in grouped[match_key]["false_answers"]:
                grouped[match_key]["false_answers"].append(false_ans)

            if mal_doc:
                grouped[match_key]["malicious_docs"].append(mal_doc)

    queries, gts, fa_groups, mdoc_groups, qids = [], [], [], [], []
    for key in order:
        data = grouped[key]
        qids.append(data["query_id"])
        queries.append(data["query"])
        gts.append(data["ground_truth_answers"])
        fa_groups.append(data["false_answers"])
        mdoc_groups.append(data["malicious_docs"])

    return queries, gts, fa_groups, mdoc_groups, qids
