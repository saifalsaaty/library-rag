"""Retrieval systems addressed by name:

  "bm25"                  lexical baseline
  "dense:<model>"         one of retrieval.dense_models in config.yaml
  "hybrid:<a>+<b>"        RRF fusion of two systems, e.g. "hybrid:bm25+bge_m3"
"""
from __future__ import annotations

import time
from pathlib import Path

from ..config import path as cfg_path
from .bm25 import BM25
from .dense import DenseEncoder, ExactIndex
from .hybrid import rrf

Run = dict[str, list[tuple[str, float]]]


def variant_dir(cfg: dict, template: str, languages: str | None = None) -> Path:
    languages = languages or cfg.get("template_languages", "all")
    return cfg_path(cfg, "index_dir") / f"{template}__{languages}"


def dense_spec(cfg: dict, name: str) -> dict:
    for m in cfg["retrieval"]["dense_models"]:
        if m["name"] == name:
            return m
    raise KeyError(f"dense model '{name}' not in config")


class SystemFactory:
    """Loads indexes once and produces runs (qid → ranked list) for named systems."""

    def __init__(self, cfg: dict, template: str | None = None, languages: str | None = None):
        self.cfg = cfg
        self.template = template or cfg["main_template"]
        self.dir = variant_dir(cfg, self.template, languages)
        self._bm25: BM25 | None = None
        self._encoders: dict[str, DenseEncoder] = {}
        self._indexes: dict[str, ExactIndex] = {}
        self._cache: dict[tuple, Run] = {}
        self.timings: dict[str, float] = {}       # seconds per query, for latency tables

    # ---------------------------------------------------------------- components
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25.load(self.dir / "bm25.pkl")
        return self._bm25

    def encoder(self, name: str) -> DenseEncoder:
        if name not in self._encoders:
            r = self.cfg["retrieval"]
            self._encoders[name] = DenseEncoder(dense_spec(self.cfg, name)["model_id"], r["device"],
                                                r["max_seq_length_cap"], r["batch_size"])
        return self._encoders[name]

    def release_gpu(self) -> None:
        """Free the encoders' GPU memory (cached runs are kept) so the LLM gets the whole card."""
        for enc in self._encoders.values():
            enc.release()
        self._encoders.clear()

    def index(self, name: str) -> ExactIndex:
        if name not in self._indexes:
            self._indexes[name] = ExactIndex.load(self.dir / "dense" / name)
        return self._indexes[name]

    # ---------------------------------------------------------------- runs
    def run(self, system: str, queries: dict[str, str], depth: int | None = None) -> Run:
        depth = depth or self.cfg["retrieval"]["depth"]
        key = (system, tuple(sorted(queries.items())), depth)
        if key in self._cache:
            return self._cache[key]
        if system == "bm25":
            bm = self.bm25()
            t0 = time.perf_counter()
            out = {qid: bm.search(q, depth) for qid, q in queries.items()}
            self.timings[system] = (time.perf_counter() - t0) / max(len(queries), 1)
        elif system.startswith("dense:"):
            name = system.split(":", 1)[1]
            enc, idx = self.encoder(name), self.index(name)
            qids = list(queries)
            t0 = time.perf_counter()
            vecs = enc.encode([queries[q] for q in qids], dense_spec(self.cfg, name)["query_prefix"])
            res = idx.search_vectors(vecs, depth)
            self.timings[system] = (time.perf_counter() - t0) / max(len(qids), 1)
            out = dict(zip(qids, res))
        elif system.startswith("hybrid:"):
            a, b = system.split(":", 1)[1].split("+")
            ra = self.run(a if a == "bm25" else f"dense:{a}", queries, self.cfg["retrieval"]["hybrid"]["fuse_depth"])
            rb = self.run(b if b == "bm25" else f"dense:{b}", queries, self.cfg["retrieval"]["hybrid"]["fuse_depth"])
            k = self.cfg["retrieval"]["hybrid"]["rrf_k"]
            out = {qid: rrf([ra.get(qid, []), rb.get(qid, [])], k=k, depth=depth) for qid in queries}
        else:
            raise ValueError(f"unknown system {system}")
        self._cache[key] = out
        return out

    def search(self, system: str, query: str, k: int) -> list[tuple[str, float]]:
        """Single-query search used by the RAG pipeline and the chat demo."""
        return self.run(system, {"q": query}, depth=max(k, self.cfg["retrieval"]["hybrid"]["fuse_depth"]))["q"][:k]


def language_variants(cfg: dict) -> list[str]:
    """Metadata-language variants for the language ablation (e.g. ['en', 'ar'])."""
    val = cfg.get("language_ablation", [])
    if val is True:
        return ["primary"]
    if not val:
        return []
    return [v for v in val if v != cfg.get("template_languages", "all")]


def all_systems(cfg: dict) -> list[str]:
    systems = ["bm25"] + [f"dense:{m['name']}" for m in cfg["retrieval"]["dense_models"]]
    systems += [f"hybrid:{a}+{b}" for a, b in cfg["retrieval"]["hybrid"]["pairs"]]
    return systems
