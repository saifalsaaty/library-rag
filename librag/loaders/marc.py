"""MARC21 loader (Koha / any ILS export).

* MARCXML (.xml) is parsed with the standard library.
* ISO 2709 (.mrc) needs `pymarc` (pip install pymarc).

Mapping (MARC21 → unified record, Dublin Core semantics):
  001            → id                 245 $a$b        → title (+ 246/880 parallel titles)
  100/110/700/710 $a → authors         600/610/650/651/653/655 → subjects
  520 $a         → abstract           505 $a          → appended to abstract as contents note
  008/35-37, 041 $a → language         260/264 $c      → year
  020 $a, 856 $u → identifiers        082 $a, 050 $a  → classification
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from ..schema import Record
from ..text import clean_text, detect_lang, normalize_lang_code

MARC_NS = {"m": "http://www.loc.gov/MARC21/slim"}
_YEAR = re.compile(r"(1[5-9]|20)\d{2}")
_TRAIL = re.compile(r"[\s/:;,.=]+$")

# generic field: (tag, value) for control fields, (tag, [(code, value), ...]) for data fields
Field = tuple


def _strip(s: str) -> str:
    return _TRAIL.sub("", clean_text(s))


def _sub(fields: list[Field], tag: str, codes: str) -> list[str]:
    out = []
    for t, val in fields:
        if t == tag and isinstance(val, list):
            parts = [_strip(v) for c, v in val if c in codes and v and _strip(v)]
            if parts:
                out.append(" ".join(parts))
    return out


def _control(fields: list[Field], tag: str) -> str:
    for t, val in fields:
        if t == tag and isinstance(val, str):
            return val
    return ""


def fields_to_record(fields: list[Field], leader: str, source: str, n: int) -> Record | None:
    rec_id = _control(fields, "001").strip() or f"rec{n:07d}"
    titles: dict[str, str] = {}
    for title in _sub(fields, "245", "abnp") + _sub(fields, "246", "ab"):
        lang = detect_lang(title) or "und"
        titles.setdefault(lang, title)
    for t, val in fields:  # 880 alternate graphic representation linked to 245
        if t == "880" and isinstance(val, list) and any(c == "6" and v.startswith("245") for c, v in val):
            title = " ".join(_strip(v) for c, v in val if c in "ab" and v)
            if title:
                titles.setdefault(detect_lang(title) or "und", title)
    if not titles:
        return None

    authors = []
    for tag in ("100", "110", "700", "710"):
        authors += _sub(fields, tag, "a")
    subjects = []
    for tag in ("600", "610", "650", "651", "653", "655"):
        for t, val in fields:
            if t == tag and isinstance(val, list):
                parts = [_strip(v) for c, v in val if c in "axyzv" and v and _strip(v)]
                if parts:
                    subjects.append(" -- ".join(parts))

    abstracts: dict[str, str] = {}
    for a in _sub(fields, "520", "a"):
        abstracts.setdefault(detect_lang(a) or "und", a)
    contents = _sub(fields, "505", "a")
    if contents:
        key = next(iter(abstracts), detect_lang(contents[0]) or "und")
        abstracts[key] = (abstracts.get(key, "") + " Contents: " + " ".join(contents)).strip()

    f008 = _control(fields, "008")
    language = normalize_lang_code(f008[35:38]) if len(f008) >= 38 else None
    if not language:
        langs = _sub(fields, "041", "a")
        language = normalize_lang_code(langs[0][:3]) if langs else detect_lang(next(iter(titles.values())))

    year = None
    for y in _sub(fields, "264", "c") + _sub(fields, "260", "c"):
        m = _YEAR.search(y)
        if m:
            year = int(m.group(0))
            break
    if year is None and len(f008) >= 11 and f008[7:11].isdigit():
        year = int(f008[7:11])

    rtype = {"a": "book", "t": "book", "m": "electronic_resource", "g": "video",
             "e": "map", "c": "score", "i": "audio", "j": "audio"}.get(leader[6:7] if leader else "", None)
    if rtype == "book" and leader[7:8] == "s":
        rtype = "serial"

    return Record(
        id=f"{source}:{rec_id}",
        source=source,
        title=titles,
        authors=list(dict.fromkeys(authors)),
        subjects=list(dict.fromkeys(subjects)),
        abstract=abstracts,
        language=language,
        year=year,
        collection=(_sub(fields, "952", "a") or [None])[0],   # Koha holding branch
        identifiers=_sub(fields, "020", "a") + _sub(fields, "856", "u"),
        rtype=rtype,
        classification=_sub(fields, "082", "a") + _sub(fields, "050", "ab"),
    )


def read_marcxml(path: str | Path, source: str) -> Iterator[Record]:
    n = 0
    for _, el in ET.iterparse(str(path), events=("end",)):
        if el.tag != f"{{{MARC_NS['m']}}}record":
            continue
        n += 1
        leader = el.findtext("m:leader", default="", namespaces=MARC_NS)
        fields: list[Field] = []
        for cf in el.findall("m:controlfield", MARC_NS):
            fields.append((cf.get("tag"), cf.text or ""))
        for df in el.findall("m:datafield", MARC_NS):
            subs = [(sf.get("code"), sf.text or "") for sf in df.findall("m:subfield", MARC_NS)]
            fields.append((df.get("tag"), subs))
        rec = fields_to_record(fields, leader, source, n)
        if rec:
            yield rec
        el.clear()


def read_iso2709(path: str | Path, source: str) -> Iterator[Record]:
    try:
        from pymarc import MARCReader  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Reading .mrc files needs pymarc:  pip install pymarc") from exc
    with open(path, "rb") as fh:
        reader = MARCReader(fh, to_unicode=True, force_utf8=True, utf8_handling="replace")
        for n, m in enumerate(reader, 1):
            if m is None:
                continue
            fields: list[Field] = []
            for f in m.fields:
                if f.is_control_field():
                    fields.append((f.tag, f.data))
                else:
                    subs = [(s.code, s.value) for s in f.subfields] if hasattr(f.subfields[0] if f.subfields else None, "code") \
                        else list(zip(f.subfields[0::2], f.subfields[1::2]))
                    fields.append((f.tag, subs))
            rec = fields_to_record(fields, str(m.leader), source, n)
            if rec:
                yield rec


def read_marc(path: str | Path, source: str) -> Iterator[Record]:
    p = Path(path)
    if p.suffix.lower() in (".xml", ".marcxml"):
        return read_marcxml(p, source)
    return read_iso2709(p, source)
