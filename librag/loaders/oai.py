"""OAI-PMH harvester for Dublin Core (oai_dc) records.

Works with OJS (Open Journal Systems), DSpace and any OAI-PMH 2.0 provider.
Every raw response page is saved to disk so the dataset can be audited and
re-parsed without re-harvesting (provenance for the Data Availability statement).
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ..schema import Record
from ..text import clean_text, detect_lang, normalize_lang_code

NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/",
    "dc": "http://purl.org/dc/elements/1.1/",
}
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
_INVALID_XML = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_YEAR = re.compile(r"(19|20)\d{2}")
_SPLIT_KW = re.compile(r"\s*[;؛،]\s*")


class OAIError(RuntimeError):
    pass


class OAIUnavailable(OAIError):
    """The endpoint answered with an HTML page (OAI disabled or behind a login)."""


def _check_not_html(content: bytes) -> None:
    head = content[:600].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise OAIUnavailable("the OAI endpoint returned an HTML page (OAI disabled or login required)")


# ------------------------------------------------------------------ parsing
def _texts(meta: ET.Element, tag: str) -> list[tuple[str, str | None]]:
    out = []
    for el in meta.findall(f"dc:{tag}", NS):
        txt = clean_text("".join(el.itertext()))
        if txt:
            out.append((txt, el.get(XML_LANG)))
    return out


def _by_lang(values: list[tuple[str, str | None]], min_chars: int = 1) -> dict[str, str]:
    out: dict[str, str] = {}
    for txt, lang in values:
        if len(txt) < min_chars:
            continue
        code = normalize_lang_code(lang) or detect_lang(txt) or "und"
        # keep the first value per language (OJS repeats the same title per locale)
        out.setdefault(code, txt)
    return out


def _rtype(types: list[str]) -> str | None:
    joined = " ".join(types).lower()
    for key, label in (("thesis", "thesis"), ("dissertation", "thesis"), ("book", "book"),
                       ("article", "article"), ("conference", "conference_paper")):
        if key in joined:
            return label
    return types[0] if types else None


def parse_record(rec: ET.Element, source: str) -> Record | None:
    """Convert one <record> element into a Record (None for deleted/empty records)."""
    header = rec.find("oai:header", NS)
    if header is None or header.get("status") == "deleted":
        return None
    meta = rec.find("oai:metadata/oai_dc:dc", NS)
    if meta is None:
        return None
    oai_id = (header.findtext("oai:identifier", default="", namespaces=NS) or "").strip()
    sets = [s.text.strip() for s in header.findall("oai:setSpec", NS) if s.text]

    titles = _by_lang(_texts(meta, "title"))
    abstracts = _by_lang(_texts(meta, "description"), min_chars=30)
    authors = list(dict.fromkeys(t for t, _ in _texts(meta, "creator")))
    subjects: list[str] = []
    for txt, _ in _texts(meta, "subject"):
        subjects.extend(s for s in _SPLIT_KW.split(txt) if len(s) > 1)
    subjects = list(dict.fromkeys(subjects))

    langs = [normalize_lang_code(t) for t, _ in _texts(meta, "language")]
    language = next((lang for lang in langs if lang), None)
    if language is None and titles:
        language = detect_lang(next(iter(titles.values())))

    year = None
    for d, _ in _texts(meta, "date"):
        m = _YEAR.search(d)
        if m:
            year = int(m.group(0))
            break

    identifiers = [t for t, _ in _texts(meta, "identifier") + _texts(meta, "relation")
                   if t.startswith("http") or t.startswith("10.")]
    sources = [t for t, _ in _texts(meta, "source")]
    collection = sets[0].split(":")[0] if sets else (sources[0] if sources else None)

    if not titles:
        return None
    return Record(
        id=f"{source}:{oai_id}",
        source=source,
        title=titles,
        authors=authors,
        subjects=subjects,
        abstract=abstracts,
        language=language,
        year=year,
        collection=collection,
        identifiers=list(dict.fromkeys(identifiers)),
        rtype=_rtype([t for t, _ in _texts(meta, "type")]),
    )


def parse_page(content: bytes, source: str) -> tuple[list[Record], str | None, int]:
    """Parse one ListRecords page → (records, resumption token, number of <record> elements)."""
    _check_not_html(content)
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        root = ET.fromstring(_INVALID_XML.sub("", content.decode("utf-8", "replace")).encode("utf-8"))
    err = root.find("oai:error", NS)
    if err is not None:
        if err.get("code") == "noRecordsMatch":
            return [], None, 0
        raise OAIError(f"{err.get('code')}: {err.text}")
    recs = root.findall("oai:ListRecords/oai:record", NS)
    out = [r for r in (parse_record(x, source) for x in recs) if r is not None]
    tok_el = root.find("oai:ListRecords/oai:resumptionToken", NS)
    token = tok_el.text.strip() if tok_el is not None and tok_el.text and tok_el.text.strip() else None
    return out, token, len(recs)


# ------------------------------------------------------------------ harvesting
def _parse_xml(content: bytes) -> ET.Element:
    _check_not_html(content)
    try:
        return ET.fromstring(content)
    except ET.ParseError:
        return ET.fromstring(_INVALID_XML.sub("", content.decode("utf-8", "replace")).encode("utf-8"))


def _get(sess, url: str, params: dict, timeout: int, max_retries: int, log, what: str):
    import requests

    for attempt in range(max_retries):
        try:
            resp = sess.get(url, params=params, timeout=timeout)
            if resp.status_code == 503:
                wait = int(resp.headers.get("Retry-After", 10 * (attempt + 1)))
                log(f"  503 from server, waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            wait = min(2 ** attempt * 5, 300)
            log(f"  request failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    raise OAIError(f"giving up after {max_retries} attempts on {what}")


def list_identifiers(sess, base_url: str, params: dict, timeout: int, max_retries: int, sleep_seconds: float,
                     log) -> list[str]:
    """All non-deleted identifiers of a repository (light requests: headers only)."""
    ids: list[str] = []
    p = {**params, "verb": "ListIdentifiers"}
    page = 0
    while True:
        page += 1
        root = _parse_xml(_get(sess, base_url, p, timeout, max_retries, log, f"ListIdentifiers page {page}").content)
        err = root.find("oai:error", NS)
        if err is not None:
            if err.get("code") == "noRecordsMatch":
                break
            raise OAIError(f"{err.get('code')}: {err.text}")
        for h in root.findall("oai:ListIdentifiers/oai:header", NS):
            if h.get("status") != "deleted":
                ident = (h.findtext("oai:identifier", default="", namespaces=NS) or "").strip()
                if ident:
                    ids.append(ident)
        tok_el = root.find("oai:ListIdentifiers/oai:resumptionToken", NS)
        token = tok_el.text.strip() if tok_el is not None and tok_el.text and tok_el.text.strip() else None
        if not token:
            break
        p = {"verb": "ListIdentifiers", "resumptionToken": token}
        time.sleep(sleep_seconds)
    return ids


def harvest(
    base_url: str,
    source: str,
    raw_dir: str | Path,
    metadata_prefix: str = "oai_dc",
    set_spec: str | None = None,
    from_date: str | None = None,
    until_date: str | None = None,
    sleep_seconds: float = 1.0,
    max_retries: int = 6,
    timeout: int = 120,
    session=None,
    log=print,
    page_dir: str | Path | None = None,
    record_fallback: bool = True,
) -> Iterator[Record]:
    """Yield Records from every ListRecords page, following resumption tokens.

    If the server keeps failing on a page (typically a single malformed record
    makes the whole batch fail with HTTP 500), the harvester falls back to
    ListIdentifiers + GetRecord and fetches the missing records one by one,
    skipping only the records the server cannot deliver.
    """
    import requests  # local import keeps parsing usable without network libs

    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", "librag-harvester/2.0 (academic research; OAI-PMH)")
    raw_dir = Path(page_dir) if page_dir else Path(raw_dir) / source / (set_spec or "all")
    raw_dir.mkdir(parents=True, exist_ok=True)

    base: dict[str, str] = {"metadataPrefix": metadata_prefix}
    if set_spec:
        base["set"] = set_spec
    if from_date:
        base["from"] = from_date
    if until_date:
        base["until"] = until_date
    params = {"verb": "ListRecords", **base}

    page, seen, got = 0, 0, set()
    mode, failed_ids = "ListRecords", []
    while True:
        page += 1
        try:
            resp = _get(sess, base_url, params, timeout, max_retries, log, f"page {page}")
        except OAIUnavailable:
            raise
        except OAIError:
            if not record_fallback:
                raise
            log(f"  page {page} keeps failing — switching to record-by-record harvesting (ListIdentifiers + GetRecord)")
            mode = "ListRecords+GetRecord"
            yield from _record_by_record(sess, base_url, base, metadata_prefix, source, raw_dir, got, failed_ids,
                                         timeout, max_retries, sleep_seconds, log)
            break
        (raw_dir / f"page_{page:05d}.xml").write_bytes(resp.content)
        records, token, n_raw = parse_page(resp.content, source)
        seen += n_raw
        log(f"  page {page}: {n_raw} OAI records, {len(records)} usable (total seen {seen})")
        for r in records:
            got.add(r.id)
        yield from records
        if not token:
            break
        params = {"verb": "ListRecords", "resumptionToken": token}
        time.sleep(sleep_seconds)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (raw_dir / "HARVEST_INFO.txt").write_text(
        f"base_url={base_url}\nset={set_spec}\nmetadataPrefix={metadata_prefix}\n"
        f"from={from_date}\nuntil={until_date}\nmode={mode}\npages={page}\nrecords_seen={seen}\n"
        f"records_harvested={len(got)}\nrecords_the_server_could_not_deliver={len(failed_ids)}\n"
        f"harvested_utc={stamp}\n",
        encoding="utf-8",
    )
    if failed_ids:
        (raw_dir / "FAILED_IDENTIFIERS.txt").write_text("\n".join(failed_ids), encoding="utf-8")


def _record_by_record(sess, base_url, base, metadata_prefix, source, raw_dir, got, failed_ids,
                      timeout, max_retries, sleep_seconds, log) -> Iterator[Record]:
    ids = list_identifiers(sess, base_url, base, timeout, max_retries, sleep_seconds, log)
    todo = [i for i in ids if f"{source}:{i}" not in got]
    log(f"  {len(ids)} identifiers listed, {len(todo)} still to fetch individually")
    rec_dir = raw_dir / "getrecord"
    rec_dir.mkdir(exist_ok=True)
    for n, ident in enumerate(todo, start=1):
        try:
            resp = _get(sess, base_url, {"verb": "GetRecord", "identifier": ident, "metadataPrefix": metadata_prefix},
                        timeout, 2, lambda *_: None, f"GetRecord {ident}")
            root = _parse_xml(resp.content)
            el = root.find("oai:GetRecord/oai:record", NS)
            rec = parse_record(el, source) if el is not None else None
        except (OAIError, ET.ParseError):
            rec = None
        if rec is None:
            failed_ids.append(ident)
        else:
            (rec_dir / f"record_{n:05d}.xml").write_bytes(resp.content)
            got.add(rec.id)
            yield rec
        if n % 50 == 0:
            log(f"    {n}/{len(todo)} fetched individually ({len(failed_ids)} could not be delivered)")
        time.sleep(sleep_seconds / 2)
    log(f"  record-by-record done: {len(todo) - len(failed_ids)} recovered, {len(failed_ids)} not deliverable")


def reparse_raw(raw_dir: str | Path, source: str) -> Iterator[Record]:
    """Re-create records from saved pages and single-record responses (no network)."""
    for p in sorted(Path(raw_dir).glob("**/page_*.xml")):
        records, _, _ = parse_page(p.read_bytes(), source)
        yield from records
    for p in sorted(Path(raw_dir).glob("**/getrecord/record_*.xml")):
        el = _parse_xml(p.read_bytes()).find("oai:GetRecord/oai:record", NS)
        rec = parse_record(el, source) if el is not None else None
        if rec is not None:
            yield rec
