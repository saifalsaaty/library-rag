"""Unified bibliographic record used by every loader, index and evaluator.

Field names follow Dublin Core semantics so that MARC21, OAI-DC and CSV
sources can be mapped onto one structure (field mappings: librag/loaders/).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class Record:
    id: str                                   # stable identifier (source:local-id)
    source: str                               # loader/source name
    title: dict[str, str] = field(default_factory=dict)      # {lang: title}
    authors: list[str] = field(default_factory=list)          # dc:creator / MARC 1XX, 7XX
    subjects: list[str] = field(default_factory=list)         # dc:subject / MARC 6XX
    abstract: dict[str, str] = field(default_factory=dict)    # {lang: text} dc:description / MARC 520
    language: str | None = None               # primary language of the resource (ar, en, ...)
    year: int | None = None
    collection: str | None = None             # journal / library branch / OAI set
    identifiers: list[str] = field(default_factory=list)      # DOI, URL, ISBN
    rtype: str | None = None                  # article, book, thesis ...
    classification: list[str] = field(default_factory=list)   # DDC / LCC (kept, not indexed)

    # ---------------------------------------------------------------
    def titles(self) -> list[str]:
        return [t for t in self.title.values() if t]

    def primary_title(self) -> str:
        if self.language and self.title.get(self.language):
            return self.title[self.language]
        for lang in ("en", "ar"):
            if self.title.get(lang):
                return self.title[lang]
        return next(iter(self.title.values()), "")

    def abstracts(self) -> list[str]:
        return [a for a in self.abstract.values() if a]

    def has(self, field_name: str) -> bool:
        if field_name == "title":
            return bool(self.titles())
        if field_name == "abstract":
            return bool(self.abstracts())
        if field_name == "authors":
            return bool(self.authors)
        if field_name == "subjects":
            return bool(self.subjects)
        raise KeyError(field_name)

    def is_bilingual(self) -> bool:
        langs = {lang for lang, t in self.title.items() if t}
        langs |= {lang for lang, a in self.abstract.items() if a}
        return len(langs & {"ar", "en"}) == 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def write_jsonl(records: Iterable[Record], p: str | Path) -> int:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(p: str | Path) -> Iterator[Record]:
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Record.from_dict(json.loads(line))
