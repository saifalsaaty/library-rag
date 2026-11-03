"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009)."""
from __future__ import annotations

from collections import defaultdict


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60, depth: int = 100) -> list[tuple[str, float]]:
    """Fuse ranked lists: score(d) = sum_i 1 / (k + rank_i(d)), ranks starting at 1."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused[:depth]
