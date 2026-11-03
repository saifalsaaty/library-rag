"""Citation extraction, catalogue verification and the verification guard.

Every resource mentioned in an answer is classified as:

  grounded                the item matches one of the records given to the model
  misattributed           the title matches a real record but the stated author does not
  catalog_not_in_context  the item exists in the catalogue but was not among the given records
  not_in_catalog          no catalogue record matches the title (fabricated or not held by the library)

The guard re-renders an answer from verified records only, so by construction it
can no longer cite anything outside the retrieved records. Its cost (lost
correct items, extra abstentions) is what the evaluation measures.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ..retrieval.bm25 import BM25
from ..schema import Record
from ..text import detect_lang, normalize, surname_tokens, title_similarity, tokenize
from .prompts import ABSTAIN_TOKEN, abstain_message

_BULLET = re.compile(r"^\s*(?:[-*•·▪–—]|\(?\d+[.)]|\(?[٠-٩]+[.)]|[a-zA-Z][.)])\s+")
_REF = re.compile(r"\[(\d+)\]")
_QUOTED = re.compile(r"[\"“”«»]([^\"“”«»]{4,})[\"“”«»]|\*\*([^*]{4,})\*\*|\*([^*]{4,})\*|_([^_]{4,})_")
# separators between title and author, tried in this order (an em dash is the format we ask for;
# " by " is tried last because it also occurs inside titles, e.g. "... by using ...")
_SPLITTERS = [re.compile(r"\s+[—–]\s+"), re.compile(r"\s+-\s+"),
              re.compile(r"\s+by\s+|\s+بقلم\s+|\s+تأليف\s+|\s+للمؤلف\s+|\s+للكاتب\s+", re.IGNORECASE)]
_FIELD_SEP = re.compile(r"\s+\|\s+")      # models sometimes echo the record format "Title: … | Authors: … | Year: …"
_AUTHOR_LABEL = re.compile(r"^\s*(?:authors?|author\(s\)|by|المؤلفون|المؤلفين|المؤلف|مؤلف|تأليف|الباحثون|الباحث)\s*[:：]\s*",
                           re.IGNORECASE)
_OTHER_FIELD = re.compile(r"^\s*(?:year|subjects?|keywords|abstract|السنة|الكلمات\s+المفتاحية|الموضوعات|الملخص)\s*[:：]",
                          re.IGNORECASE)
_MD = re.compile(r"[*_`#]+")
_LABEL = re.compile(r"^(title|العنوان)\s*:\s*", re.IGNORECASE)
# commentary lines that carry a [n] or a bullet but recommend nothing ("[1] is the most relevant …",
# "Both records are relevant …", "لا أجد أي سجل …")
_COMMENTARY = re.compile(r"^\s*(?:(?:is|are|was|seems?)\b|both\b|these\b|this record\b|the records?\b|none of\b|all (?:of )?the records?\b|"
                         r"كلا\b|هذه السجلات|هذا السجل|لا أجد|لا يوجد|لا توجد|لم أجد)", re.IGNORECASE)
REF_ASSISTED_THRESHOLD = 0.70   # a [n] citation plus a fairly similar title identifies the cited record
# words that show a stretch of text is a sentence (an explanation), not a list of names
_SENTENCE_WORDS = {normalize(w) for w in (
    "this", "the", "is", "are", "was", "record", "relevant", "because", "paper", "study", "discusses", "about",
    "which", "it", "its", "provides", "explores", "examines", "focuses", "related", "useful", "research",
    "هذا", "هذه", "يتناول", "يناقش", "البحث", "الدراسة", "لأنه", "حيث", "التي", "الذي", "يقدم", "يركز", "مفيد", "ذات",
)}

_ABSTAIN_PHRASES = [normalize(p) for p in (
    "no relevant", "don't have relevant", "do not have relevant", "not have any relevant", "couldn't find",
    "could not find", "unable to find", "none of the records", "not in the catalog", "no records",
    "لا توجد", "لا يوجد", "لم أجد", "لا تتوفر", "لا تتضمن", "ليس لدي", "لا تحتوي", "لا تتعلق",
)]


@dataclass
class Item:
    line: str
    refs: list[int]
    candidates: list[str]
    author: str
    reason: str
    title: str = ""
    label: str = ""
    record_id: str | None = None
    similarity: float = 0.0
    citation_mismatch: bool = False


@dataclass
class Verification:
    abstained: bool
    items: list[Item] = field(default_factory=list)

    def counts(self) -> dict:
        labels = [i.label for i in self.items]
        out = {lab: labels.count(lab) for lab in ("grounded", "misattributed", "catalog_not_in_context", "not_in_catalog")}
        out["n_items"] = len(self.items)
        out["citation_mismatch"] = sum(i.citation_mismatch for i in self.items)
        return out

    def to_dict(self) -> dict:
        return {"abstained": self.abstained, **self.counts(), "items": [asdict(i) for i in self.items]}


# ------------------------------------------------------------------ parsing
def _clean(s: str) -> str:
    s = _MD.sub("", _REF.sub("", s))
    s = _LABEL.sub("", s.strip())
    return s.strip(" \t:;,.\"'“”«»()[]")


def _split_body(body: str) -> tuple[str, str, str]:
    """Split one answer line into (title part, stated authors, explanation)."""
    if _FIELD_SEP.search(body):                       # record-style echo
        fields = _FIELD_SEP.split(body)
        title_part, author, reason = fields[0], "", []
        for f in fields[1:]:
            if _AUTHOR_LABEL.match(f):
                author = _AUTHOR_LABEL.sub("", f)
            elif not _OTHER_FIELD.match(f):
                reason.append(f)
        inner_t, inner_a, inner_r = _split_body(title_part)    # "Title — Authors: …" before the first "|"
        return inner_t, author or inner_a, " ".join([inner_r, *reason]).strip()
    for splitter in _SPLITTERS:
        parts = splitter.split(body, maxsplit=1)
        if len(parts) == 2:                          # "Title — Author(s): reason"
            title_part, rest = parts
            rest = _AUTHOR_LABEL.sub("", rest)          # "Title — Authors: A; B: reason"
            author, _, reason = rest.partition(":")
            return title_part, author, reason
    return body, "", body.partition(": ")[2]           # "Title: reason" or a bare title


def _looks_like_names(s: str) -> bool:
    toks = tokenize(s)
    return 0 < len(toks) <= 16 and not any(t in _SENTENCE_WORDS for t in toks)


def parse_items(text: str) -> list[Item]:
    """Extract resource mentions from a free-text answer.

    A line counts as an item if it is a list entry, carries a [n] reference, or
    contains a quoted / emphasised title. Introductory sentences are ignored.
    """
    items: list[Item] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.endswith(":") and not _REF.search(line):
            continue
        is_bullet = bool(_BULLET.match(line))
        refs = [int(x) for x in _REF.findall(line)]
        quoted = [next(g for g in m.groups() if g) for m in _QUOTED.finditer(line)]
        if not (is_bullet or refs or quoted):
            continue
        body = _REF.sub("", _BULLET.sub("", line, count=1)).strip()
        if ABSTAIN_TOKEN in body or _COMMENTARY.match(_MD.sub("", body)):
            continue
        title_part, author, reason = _split_body(body)
        reason = reason.strip()
        candidates = [_clean(q) for q in quoted]
        candidates += [_clean(title_part), _clean(title_part.split(": ")[0])]
        candidates += [_clean(piece) for piece in title_part.split(" / ")]      # "Arabic title / English title"
        candidates = [c for c in dict.fromkeys(candidates) if len(tokenize(c)) >= 2]
        if not candidates and not refs:
            continue
        items.append(Item(line=line, refs=refs, candidates=candidates, author=_clean(author), reason=reason[:400]))
    return items


def is_abstention(text: str, items: list[Item]) -> bool:
    if ABSTAIN_TOKEN in text:
        return not any(i.refs or i.candidates for i in items) or len(items) == 0
    if items:
        return False
    norm = normalize(text)
    return any(p in norm for p in _ABSTAIN_PHRASES) or not norm.strip()


# ------------------------------------------------------------------ matching
class CatalogMatcher:
    """Finds the best catalogue record for a title string (BM25 shortlist → fuzzy similarity)."""

    def __init__(self, records: dict[str, Record], candidate_pool: int = 30):
        self.records = records
        ids = list(records)
        self.index = BM25(k1=1.2, b=0.75, stem=False, remove_stopwords=False).fit(
            ids, [" ".join(records[i].titles()) for i in ids])
        self.pool = candidate_pool

    def best(self, title: str, restrict: list[str] | None = None) -> tuple[str | None, float]:
        ids = restrict if restrict is not None else [d for d, _ in self.index.search(title, self.pool)]
        best_id, best_sim = None, 0.0
        for rid in ids:
            for t in self.records[rid].titles():
                s = title_similarity(title, t)
                if s > best_sim:
                    best_id, best_sim = rid, s
        return best_id, best_sim


def _author_agrees(author: str, rec: Record) -> bool | None:
    """True if a stated author matches the record, False if none does, None when it cannot be judged.

    Each stated name is compared only with the record's authors written in the same script
    (an Arabic name cannot be checked automatically against a romanised one). Text that reads
    like a sentence rather than a list of names (e.g. the explanation written where the
    authors should be) is not judged.
    """
    if not author or not rec.authors or not _looks_like_names(author):
        return None
    verdicts = []
    for name in re.split(r"\s*(?:;|؛|&|\band\b)\s*", author):
        stated = surname_tokens(name)
        script = detect_lang(name)
        same_script = [a for a in rec.authors if detect_lang(a) == script]
        if not stated or not same_script:
            continue
        known = set().union(*(surname_tokens(a) for a in same_script))
        verdicts.append(bool(stated & known))
    return any(verdicts) if verdicts else None


def verify(text: str, context_ids: list[str], matcher: CatalogMatcher, threshold: float = 0.85) -> Verification:
    items = parse_items(text)
    result = Verification(abstained=is_abstention(text, items))
    if result.abstained:
        return result
    for it in items:
        ref_ids = [context_ids[r - 1] for r in it.refs if 1 <= r <= len(context_ids)]
        best_ctx, sim_ctx, best_cat, sim_cat, best_title = None, 0.0, None, 0.0, ""
        for cand in it.candidates or [""]:
            if cand:
                rid, s = matcher.best(cand, restrict=context_ids) if context_ids else (None, 0.0)
                if s > sim_ctx:
                    best_ctx, sim_ctx, best_title = rid, s, cand
                rid2, s2 = matcher.best(cand)
                if s2 > sim_cat:
                    best_cat, sim_cat = rid2, s2
                    best_title = best_title or cand
        it.title = best_title or (it.candidates[0] if it.candidates else "")
        if sim_ctx >= threshold or (best_ctx in ref_ids and sim_ctx >= REF_ASSISTED_THRESHOLD):
            it.label, it.record_id, it.similarity = "grounded", best_ctx, sim_ctx
            it.citation_mismatch = bool(ref_ids) and best_ctx not in ref_ids
        elif ref_ids and not it.candidates:
            it.label, it.record_id, it.similarity = "grounded", ref_ids[0], 1.0   # bare [n] citation
        elif sim_cat >= threshold:
            it.label, it.record_id, it.similarity = "catalog_not_in_context", best_cat, sim_cat
        else:
            it.label, it.record_id, it.similarity = "not_in_catalog", best_cat, max(sim_ctx, sim_cat)
        if it.label in ("grounded", "catalog_not_in_context") and it.record_id:
            if _author_agrees(it.author, matcher.records[it.record_id]) is False:
                it.label = "misattributed"
        result.items.append(it)
    return result


def guard(v: Verification, context_ids: list[str], records: dict[str, Record], lang: str | None) -> str:
    """Verification layer: rebuild the answer from verified context records only."""
    lines = []
    seen = set()
    for it in v.items:
        if it.label != "grounded" or it.record_id in seen:
            continue
        seen.add(it.record_id)
        r = records[it.record_id]
        n = context_ids.index(it.record_id) + 1 if it.record_id in context_ids else "?"
        authors = "; ".join(r.authors[:3]) or ("مؤلف غير مذكور" if lang == "ar" else "author not recorded")
        year = f" ({r.year})" if r.year else ""
        reason = f": {it.reason}" if it.reason else ""
        lines.append(f"- [{n}] {r.primary_title()} — {authors}{year}{reason}")
    return "\n".join(lines) if lines else abstain_message(lang)
