"""Inter-annotator agreement for relevance and hallucination judgements."""
from __future__ import annotations

import numpy as np


def cohen_kappa(a, b, weights: str | None = None) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b, weights=weights))


def fleiss_kappa(ratings: np.ndarray) -> float:
    """ratings: items x categories matrix of counts (each row sums to the number of raters)."""
    r = np.asarray(ratings, float)
    n_items, _ = r.shape
    n_raters = r.sum(axis=1)[0]
    p_j = r.sum(axis=0) / (n_items * n_raters)
    p_i = ((r * r).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else 1.0


def percent_agreement(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float((a == b).mean()) if a.size else float("nan")


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) bands."""
    if k < 0:
        return "poor"
    for limit, label in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= limit:
            return label
    return "almost perfect"


def double_judged_queries(queries, fraction: float, seed: int, strata=("lang", "type", "answerable")) -> list[str]:
    """Stratified random sample of query ids that every annotator judges.

    The remaining queries are judged by the first annotator only. Sampling whole
    queries (not single pairs) keeps each annotator's view of a request complete.
    With fraction >= 1 every query is returned.
    """
    import math
    import random

    qids = list(queries["qid"])
    if fraction >= 1:
        return qids
    rng = random.Random(seed)
    chosen: list[str] = []
    for _, g in queries.groupby(list(strata), sort=True):
        ids = sorted(g["qid"])
        k = min(len(ids), max(1, math.floor(len(ids) * fraction + 0.5)))
        chosen.extend(rng.sample(ids, k))
    return [q for q in qids if q in set(chosen)]


def combine_judgements(frames: dict, first: str):
    """Merge per-annotator grades (each a Series indexed by (qid, doc_id)).

    Returns (overlap, final): `overlap` holds the pairs judged by every annotator
    (used for agreement and adjudication); `final` holds one grade per judged pair:
    the lower grade where annotators disagree (until adjudicated) and the first
    annotator's grade for pairs only that annotator judged.
    """
    import pandas as pd

    J = pd.DataFrame(frames)
    overlap = J.dropna()
    final = J[first].copy()
    final.loc[overlap.index] = overlap.min(axis=1)
    return overlap.astype(int), final.dropna().astype(int)
