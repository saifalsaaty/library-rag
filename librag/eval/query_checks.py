"""Checks on the test queries before pooling.

A query that repeats a record title word for word makes retrieval artificially easy
and favours lexical search (guidelines, section 1, rules 2 and 4). These checks find
such queries before any annotation time is spent on them.
"""
from __future__ import annotations

import pandas as pd

from ..schema import Record
from ..text import tokenize


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def longest_common_run(a: list[str], b: list[str]) -> int:
    """Length of the longest run of consecutive words shared by a and b."""
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def copied_titles(queries: pd.DataFrame, records: dict[str, Record], n: int = 6) -> pd.DataFrame:
    """One row per query that repeats at least `n` consecutive words of some record title.

    Columns: qid, words_copied, n_titles (how many titles share the run), doc_id, title (the closest one).
    """
    index: dict[tuple[str, ...], set[str]] = {}
    for rid, r in records.items():
        for t in r.titles():
            for g in _ngrams(tokenize(t), n):
                index.setdefault(g, set()).add(rid)
    rows = []
    for qid, row in queries.iterrows():
        qt = tokenize(str(row["query"]))
        hits: set[str] = set()
        for g in _ngrams(qt, n):
            hits |= index.get(g, set())
        if not hits:
            continue
        best = max(((longest_common_run(qt, tokenize(t)), rid, t) for rid in hits for t in records[rid].titles()),
                   key=lambda x: x[0])
        rows.append({"qid": qid, "words_copied": best[0], "n_titles": len(hits), "doc_id": best[1], "title": best[2]})
    return pd.DataFrame(rows, columns=["qid", "words_copied", "n_titles", "doc_id", "title"])
