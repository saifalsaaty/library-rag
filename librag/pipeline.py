"""End-to-end RAG pipeline: retrieve → prompt → local LLM → verify (→ guard).

The same object is used by the experiments (scripts/07_run_generation.py) and
by the interactive demo (scripts/chat_demo.py), so what is evaluated is
exactly what is demonstrated.
"""
from __future__ import annotations

import time

from .generation.prompts import CONDITIONS, build_messages
from .generation.verify import CatalogMatcher, guard, verify
from .schema import Record
from .text import detect_lang


class RAGPipeline:
    def __init__(self, records: dict[str, Record], llm, retriever=None, retriever_name: str = "",
                 matcher: CatalogMatcher | None = None, threshold: float = 0.85, max_abstract_chars: int = 600):
        """retriever: object with .search(system_name, query, k) (SystemFactory) or None for no-RAG runs."""
        self.records = records
        self.llm = llm
        self.retriever = retriever
        self.retriever_name = retriever_name
        self.matcher = matcher or CatalogMatcher(records)
        self.threshold = threshold
        self.max_abstract_chars = max_abstract_chars

    def answer(self, query: str, condition: str = "rag_grounded", k: int = 3, lang: str | None = None) -> dict:
        _, uses_retrieval = CONDITIONS[condition]
        context_ids: list[str] = []
        retrieval_s = 0.0
        if uses_retrieval:
            t0 = time.perf_counter()
            context_ids = [d for d, _ in self.retriever.search(self.retriever_name, query, k)]
            retrieval_s = time.perf_counter() - t0
        context = [self.records[d] for d in context_ids]
        system, user = build_messages(query, condition, context, self.max_abstract_chars)
        out = self.llm.chat(system, user)
        v = verify(out["text"], context_ids, self.matcher, self.threshold)
        lang = lang or detect_lang(query)
        guarded = guard(v, context_ids, self.records, lang) if uses_retrieval else None
        return {
            "condition": condition,
            "k": k if uses_retrieval else 0,
            "context_ids": context_ids,
            "retrieval_s": round(retrieval_s, 4),
            "response": out["text"],
            "guarded_response": guarded,
            "llm": {key: val for key, val in out.items() if key != "text"},
            "verification": v.to_dict(),
        }
