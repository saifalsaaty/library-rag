"""Cleaning, de-duplication, field templates and descriptive statistics."""
from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np

from .schema import Record
from .text import AR_STOPWORDS, EN_STOPWORDS, clean_text, detect_lang, normalize, title_similarity, tokenize

FIELD_LABELS = {"title": "Title", "authors": "Authors", "subjects": "Subjects", "abstract": "Abstract"}


# ------------------------------------------------------------------ cleaning
_KW_PREFIX = re.compile(r"^\s*(?:key\s*-?\s*words?|index\s+terms|الكلمات\s+(?:المفتاحية|الدالة)|كلمات\s+(?:مفتاحية|دالة))\s*[:：\-]?\s*",
                        re.IGNORECASE)
_KW_SPLIT = re.compile(r"\s*(?:,,+|;|؛|،)\s*")
_SUBJECT_JUNK = {normalize(x) for x in (
    "[]", "{}", "()", "العدد الحالي", "current issue", "research article", "research paper", "article",
    "articles", "original article", "review article", "مقالة بحثية", "بحث", "main subjects",
    "المواضيع الرئيسية", "المحاور الرئيسية",
)}


def clean_subjects(values: list[str]) -> list[str]:
    """Remove platform artefacts ('[]', 'العدد الحالي'), 'Key words:' prefixes and split packed lists."""
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        v = _KW_PREFIX.sub("", clean_text(v))
        parts = _KW_SPLIT.split(v) if _KW_SPLIT.search(v) else [v]
        for p in parts:
            p = p.strip(" ,.;:-–[](){}\"'")
            key = normalize(p)
            if len(p) < 2 or key in _SUBJECT_JUNK or key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


_FRONT_MATTER = {normalize(x) for x in (
    "editorial", "foreword", "preface", "special issue", "contents", "table of contents", "index",
    "erratum", "corrigendum", "retraction", "instructions to authors", "guide for authors",
    "محتويات العدد", "الافتتاحية", "كلمة العدد", "كلمة التحرير", "المحتويات", "الفهرس", "تقديم",
)}


def is_front_matter(r: Record) -> bool:
    """Editorials, contents pages etc. are not retrievable research records."""
    return any(normalize(t).strip(" .:-") in _FRONT_MATTER for t in r.titles())


def clean_record(r: Record, placeholders: set[str], min_title_chars: int) -> Record | None:
    def ok(s: str) -> bool:
        return bool(s) and normalize(s).strip(" .-") not in placeholders

    r.title = {k: clean_text(v) for k, v in r.title.items() if ok(clean_text(v))}
    r.title = {k: v for k, v in r.title.items() if len(v) >= min_title_chars}
    if not r.title or is_front_matter(r):
        return None
    r.abstract = {k: clean_text(v) for k, v in r.abstract.items() if ok(clean_text(v))}
    r.authors = [clean_text(a) for a in r.authors if ok(clean_text(a))]
    r.subjects = [x for x in clean_subjects(r.subjects) if ok(x)]
    return r


def _dedup_key(r: Record) -> str:
    """Same normalised title in the same collection = one work (migrated archives list some papers twice)."""
    title = " ".join(tokenize(r.title.get("en") or r.primary_title()))
    return f"{r.collection or ''}|{title}"


def _richness(r: Record) -> int:
    return sum(r.has(f) for f in ("title", "authors", "subjects", "abstract")) * 10 + len(r.titles()) + len(r.abstracts())


def deduplicate(records: Iterable[Record]) -> tuple[list[Record], int]:
    """Keep the richest record among those sharing an identifier, or a normalised title within one collection."""
    best: dict[str, Record] = {}
    ident_to_key: dict[str, str] = {}
    n_in = 0
    for r in records:
        n_in += 1
        key = _dedup_key(r)
        for ident in r.identifiers:
            if ident in ident_to_key:
                key = ident_to_key[ident]
                break
        if key not in best or _richness(r) > _richness(best[key]):
            best[key] = r
        for ident in r.identifiers:
            ident_to_key.setdefault(ident, key)
    unique = list(best.values())
    return unique, n_in - len(unique)


def stratified_sample(records: list[Record], n: int, seed: int) -> list[Record]:
    by_col: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_col[r.collection or "_"].append(r)
    rng = random.Random(seed)
    out: list[Record] = []
    for col, items in sorted(by_col.items()):
        share = max(1, round(n * len(items) / len(records)))
        out.extend(rng.sample(items, min(share, len(items))))
    rng.shuffle(out)
    return out[:n]


def stable_id(r: Record) -> str:
    return hashlib.sha1(r.id.encode("utf-8")).hexdigest()[:12]


def find_by_title(records: dict[str, Record], title: str, threshold: float = 0.85) -> tuple[str | None, float]:
    """Return (record id, similarity) of the record whose Arabic or English title best matches `title`.

    Used to locate the intended work of a known-item query. Returns (None, best score)
    when no title reaches `threshold`. Candidates are pre-filtered by shared words.
    """
    words = {t for t in tokenize(title) if len(t) > 2}
    if not words:
        return None, 0.0
    best_id, best = None, 0.0
    for rid, r in records.items():
        for t in r.titles():
            tw = set(tokenize(t))
            if len(words & tw) < max(1, len(words) // 3):
                continue
            sim = title_similarity(title, t)
            if sim > best:
                best_id, best = rid, sim
    return (best_id, best) if best >= threshold else (None, best)


def _longest_shared_run(a: list[str], b: list[str]) -> list[str]:
    """Longest run of consecutive tokens that appears in both token lists."""
    best, end = 0, 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], i
        prev = cur
    return a[end - best:end]


def copied_phrases(records: dict[str, Record], query: str, min_content_words: int = 4,
                   top: int = 3) -> list[tuple[int, str, str, str]]:
    """Find titles or keyword lists that share a run of >= `min_content_words` content words with `query`.

    Returns up to `top` hits as (content words in the run, record id, field, shared phrase).
    Used to screen test queries for phrases copied from records, which would make
    retrieval artificially easy for lexical systems.
    """
    stop = EN_STOPWORDS | AR_STOPWORDS
    q = tokenize(query)
    q_content = {w for w in q if w not in stop and len(w) > 1}
    if len(q_content) < min_content_words:
        return []
    hits = []
    for rid, r in records.items():
        fields = [("title", t) for t in r.titles()] + ([("subjects", " ".join(r.subjects))] if r.subjects else [])
        for name, text in fields:
            toks = tokenize(text)
            if len(q_content & set(toks)) < min_content_words:
                continue
            run = _longest_shared_run(q, toks)
            n = sum(1 for w in run if w not in stop and len(w) > 1)
            if n >= min_content_words:
                hits.append((n, rid, name, " ".join(run)))
    hits.sort(key=lambda h: -h[0])
    return hits[:top]


# ------------------------------------------------------------------ templates
def record_text(r: Record, fields: list[str], languages: str = "all") -> str:
    """Render a record as labelled text, e.g. 'Title: ... | Subjects: ... | Abstract: ...'.

    languages:
      'all'      parallel Arabic and English metadata (default)
      'en'/'ar'  only metadata in that language, falling back to whatever exists
                 when a record has none in it (language ablation, RQ2)
      'primary'  only the language given in the record's language field
    """
    def pick(d: dict[str, str]) -> list[str]:
        if languages == "primary" and r.language and r.language in d:
            return [d[r.language]]
        if languages in ("ar", "en"):
            if d.get(languages):
                return [d[languages]]
        return [v for v in d.values() if v]

    def pick_subjects() -> list[str]:
        if languages in ("ar", "en"):
            same = [x for x in r.subjects if detect_lang(x) in (languages, None)]
            if same:
                return same
        return r.subjects

    parts = []
    for f in fields:
        if f == "title":
            vals = pick(r.title)
            if vals:
                parts.append(f"{FIELD_LABELS[f]}: " + " / ".join(vals))
        elif f == "abstract":
            vals = pick(r.abstract)
            if vals:
                parts.append(f"{FIELD_LABELS[f]}: " + " ".join(vals))
        elif f == "authors" and r.authors:
            parts.append(f"{FIELD_LABELS[f]}: " + "; ".join(r.authors))
        elif f == "subjects" and r.subjects:
            parts.append(f"{FIELD_LABELS[f]}: " + "; ".join(pick_subjects()))
    return " | ".join(parts)


# ------------------------------------------------------------------ statistics
def describe(records: list[Record]) -> dict:
    """Descriptive statistics reported in the Data section of the paper."""
    n = len(records)
    if n == 0:
        return {"n_records": 0}

    def pct(x):
        return round(100.0 * x / n, 1)

    def length_stats(values):
        values = [v for v in values if v > 0]
        if not values:
            return {"n_nonzero": 0, "mean": 0.0, "median": 0.0, "p90": 0.0}
        arr = np.asarray(values, dtype=float)
        return {"n_nonzero": int(arr.size), "mean": round(float(arr.mean()), 1),
                "median": round(float(np.median(arr)), 1), "p90": round(float(np.percentile(arr, 90)), 1)}

    lang = Counter(r.language or "unknown" for r in records)
    coll = Counter(r.collection or "unknown" for r in records)
    rtype = Counter(r.rtype or "unknown" for r in records)
    years = [r.year for r in records if r.year]
    codes = {"title": "T", "authors": "Au", "subjects": "S", "abstract": "Ab"}
    combos = Counter("+".join(c for f, c in codes.items() if r.has(f)) for r in records)
    title_langs = Counter("+".join(sorted(k for k, v in r.title.items() if v)) for r in records)
    return {
        "n_records": n,
        "language_distribution (record language field)": dict(lang.most_common()),
        "title_languages": dict(title_langs.most_common()),
        "has_arabic_title_pct": pct(sum(1 for r in records if r.title.get("ar"))),
        "has_english_title_pct": pct(sum(1 for r in records if r.title.get("en"))),
        "collections": dict(coll.most_common()),
        "record_types": dict(rtype.most_common()),
        "bilingual_ar_en_pct": pct(sum(r.is_bilingual() for r in records)),
        "field_completeness_pct": {f: pct(sum(r.has(f) for r in records))
                                   for f in ("title", "authors", "subjects", "abstract")},
        "field_combinations": dict(combos.most_common()),
        "title_words": length_stats([len(tokenize(r.primary_title())) for r in records]),
        "abstract_words": length_stats([sum(len(tokenize(a)) for a in r.abstracts()) for r in records]),
        "subjects_per_record": length_stats([len(r.subjects) for r in records]),
        "authors_per_record": length_stats([len(r.authors) for r in records]),
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
        "year_missing_pct": pct(n - len(years)),
    }
