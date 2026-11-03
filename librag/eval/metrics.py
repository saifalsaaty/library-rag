"""Ranking metrics with trec_eval conventions.

* Unjudged documents count as non-relevant.
* P@k divides by k even when fewer than k documents are returned.
* nDCG uses linear gain (relevance grade) and log2(rank + 1) discount.
* Only queries with at least one relevant judgement are evaluated.
* judged@k reports pool coverage (share of the top-k that was judged), which
  must be reported for systems that did not contribute to the pool.
"""
from __future__ import annotations

import math

import pandas as pd

Run = dict[str, list[tuple[str, float]]]
Qrels = dict[str, dict[str, int]]


def _ranked(run_q: list[tuple[str, float]]) -> list[str]:
    return [d for d, _ in run_q]


def precision_at(ranked: list[str], rels: dict[str, int], k: int) -> float:
    return sum(1 for d in ranked[:k] if rels.get(d, 0) > 0) / k


def recall_at(ranked: list[str], rels: dict[str, int], k: int) -> float:
    n_rel = sum(1 for v in rels.values() if v > 0)
    return sum(1 for d in ranked[:k] if rels.get(d, 0) > 0) / n_rel if n_rel else 0.0


def success_at(ranked: list[str], rels: dict[str, int], k: int) -> float:
    return 1.0 if any(rels.get(d, 0) > 0 for d in ranked[:k]) else 0.0


def rr_at(ranked: list[str], rels: dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if rels.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if rels.get(d, 0) > 0)
    ideal = sorted((v for v in rels.values() if v > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def average_precision(ranked: list[str], rels: dict[str, int]) -> float:
    n_rel = sum(1 for v in rels.values() if v > 0)
    if not n_rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if rels.get(d, 0) > 0:
            hits += 1
            total += hits / i
    return total / n_rel


def judged_at(ranked: list[str], rels: dict[str, int], k: int) -> float:
    top = ranked[:k]
    return sum(1 for d in top if d in rels) / k


def per_query(run: Run, qrels: Qrels, cutoffs=(1, 3, 5, 10)) -> pd.DataFrame:
    """One row per evaluated query; columns are metrics (ndcg@10, mrr@10, p@5, r@10, map, ...)."""
    rows = []
    for qid, rels in qrels.items():
        if not any(v > 0 for v in rels.values()):
            continue
        ranked = _ranked(run.get(qid, []))
        row = {"qid": qid, "map": average_precision(ranked, rels), "n_retrieved": len(ranked)}
        for k in cutoffs:
            row[f"ndcg@{k}"] = ndcg_at(ranked, rels, k)
            row[f"p@{k}"] = precision_at(ranked, rels, k)
            row[f"r@{k}"] = recall_at(ranked, rels, k)
            row[f"success@{k}"] = success_at(ranked, rels, k)
            row[f"judged@{k}"] = judged_at(ranked, rels, k)
        row[f"mrr@{max(cutoffs)}"] = rr_at(ranked, rels, max(cutoffs))
        row["r@100"] = recall_at(ranked, rels, 100)
        rows.append(row)
    return pd.DataFrame(rows).set_index("qid").sort_index() if rows else pd.DataFrame()


def zero_result_rate(run: Run, qids: list[str]) -> float:
    return sum(1 for q in qids if not run.get(q)) / len(qids) if qids else 0.0


def condense(run: Run, qrels: Qrels, complete: set[str] | frozenset = frozenset()) -> Run:
    """Condensed-list run (Sakai, 2007): unjudged documents are removed before scoring.

    Used for runs that did not contribute to the judgement pool, whose unjudged documents
    would otherwise all count as non-relevant. Queries in `complete` (e.g. known-item
    queries, where every record except the target is non-relevant by definition) are kept as is.
    """
    out: Run = {}
    for qid, ranked in run.items():
        if qid in complete:
            out[qid] = ranked
        else:
            rels = qrels.get(qid, {})
            out[qid] = [(d, s) for d, s in ranked if d in rels]
    return out
