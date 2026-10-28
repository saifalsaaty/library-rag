"""Okapi BM25 (lexical baseline) implemented on a sparse matrix.

idf = ln(1 + (N - df + 0.5) / (df + 0.5))  (Lucene variant, always positive)
Defaults k1 = 0.9, b = 0.4 follow the Anserini/Pyserini reference settings.
Unlike dense retrieval, BM25 can legitimately return zero results, which is
what makes the zero-result rate a meaningful comparison.
"""
from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from ..text import analyze


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4, stem: bool = True, remove_stopwords: bool = True):
        self.k1, self.b, self.stem, self.remove_stopwords = k1, b, stem, remove_stopwords
        self.vocab: dict[str, int] = {}
        self.doc_ids: list[str] = []
        self.W: sparse.csr_matrix | None = None
        self.idf: np.ndarray | None = None

    def _analyze(self, text: str) -> list[str]:
        return analyze(text, stem=self.stem, remove_stopwords=self.remove_stopwords)

    def fit(self, doc_ids: list[str], texts: list[str]) -> "BM25":
        rows, cols, vals = [], [], []
        lengths = np.zeros(len(texts), dtype=np.float64)
        for i, t in enumerate(texts):
            counts = Counter(self._analyze(t))
            lengths[i] = sum(counts.values())
            for term, c in counts.items():
                j = self.vocab.setdefault(term, len(self.vocab))
                rows.append(i)
                cols.append(j)
                vals.append(c)
        n, v = len(texts), len(self.vocab)
        tf = sparse.csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(n, v))
        df = np.bincount(np.asarray(cols, dtype=np.int64), minlength=v).astype(np.float64)
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5))
        avgdl = lengths.mean() if n else 0.0
        norm = self.k1 * (1.0 - self.b + self.b * lengths / (avgdl or 1.0))
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1.0) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=(n, v))
        self.doc_ids = list(doc_ids)
        return self

    def query_vector(self, query: str) -> np.ndarray:
        q = np.zeros(len(self.vocab), dtype=np.float64)
        for term, c in Counter(self._analyze(query)).items():
            j = self.vocab.get(term)
            if j is not None:
                q[j] = c
        return q

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.W @ self.query_vector(query)).ravel()

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        s = self.scores(query)
        nz = np.flatnonzero(s > 0)
        if nz.size == 0:
            return []
        order = nz[np.lexsort((nz, -s[nz]))][:k]       # score desc, then index asc (deterministic)
        return [(self.doc_ids[i], float(s[i])) for i in order]

    def save(self, p: str | Path) -> None:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, p: str | Path) -> "BM25":
        obj = cls()
        with open(p, "rb") as fh:
            obj.__dict__.update(pickle.load(fh))
        return obj
