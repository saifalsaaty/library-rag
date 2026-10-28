"""Generic CSV/Excel loader with a configurable column mapping."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ..schema import Record
from ..text import clean_text, detect_lang, normalize_lang_code


def read_table(path: str | Path, source: str, columns: dict[str, str], list_separator: str = ";") -> Iterator[Record]:
    p = Path(path)
    df = pd.read_excel(p, dtype=str) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p, dtype=str, encoding="utf-8-sig")
    df = df.fillna("")

    def col(row, key):
        c = columns.get(key)
        return clean_text(row[c]) if c and c in row else ""

    def split(s):
        return [x.strip() for x in s.split(list_separator) if x.strip()] if s else []

    for i, row in df.iterrows():
        title = col(row, "title")
        if not title:
            continue
        abstract = col(row, "abstract")
        lang = normalize_lang_code(col(row, "language")) or detect_lang(title)
        year_s = col(row, "year")
        yield Record(
            id=f"{source}:{col(row, 'id') or i}",
            source=source,
            title={detect_lang(title) or "und": title},
            authors=split(col(row, "authors")),
            subjects=split(col(row, "subjects")),
            abstract={detect_lang(abstract) or "und": abstract} if abstract else {},
            language=lang,
            year=int(year_s[:4]) if year_s[:4].isdigit() else None,
            collection=col(row, "collection") or None,
        )
