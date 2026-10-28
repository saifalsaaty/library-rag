"""Prompts for the 2 x 2 design: {no retrieval, retrieval} x {plain, grounded} instructions.

Both prompts ask for the SAME output format so that citation extraction is
equally reliable in every condition; only the grounding rules differ.
"""
from __future__ import annotations

from ..schema import Record

ABSTAIN_TOKEN = "NO_RELEVANT_RECORDS"

PLAIN_SYSTEM = (
    "You are a helpful academic librarian at a university library.\n"
    "Recommend library resources (books, articles or theses) that answer the user's request.\n"
    "Reply in the same language as the user's request.\n"
    "List each recommended resource on its own line in this exact format:\n"
    "- Title — Author(s): one sentence explaining why it is relevant."
)

GROUNDED_SYSTEM = (
    "You are an academic librarian at a university library.\n"
    "You may recommend ONLY resources that appear in the numbered catalogue records given by the user.\n"
    "Rules:\n"
    "1. Never mention any book, article, thesis or author that is not in the records.\n"
    "2. Copy each title exactly as written in the record and cite its record number in square brackets.\n"
    "3. Recommend only records that are relevant to the request.\n"
    f"4. If none of the records is relevant, reply with exactly: {ABSTAIN_TOKEN}\n"
    "Reply in the same language as the user's request.\n"
    "Format, one line per resource:\n"
    "- [n] Title — Author(s): one sentence explaining why it is relevant."
)

CONDITIONS = {
    #  name             (system prompt,   uses retrieval)
    "norag_plain":    (PLAIN_SYSTEM, False),
    "norag_grounded": (GROUNDED_SYSTEM, False),
    "rag_plain":      (PLAIN_SYSTEM, True),
    "rag_grounded":   (GROUNDED_SYSTEM, True),
}


def format_record(n: int, r: Record, max_abstract_chars: int = 600) -> str:
    parts = [f"[{n}] Title: " + " / ".join(r.titles())]
    if r.authors:
        parts.append("Authors: " + "; ".join(r.authors[:6]))
    if r.year:
        parts.append(f"Year: {r.year}")
    if r.subjects:
        parts.append("Subjects: " + "; ".join(r.subjects[:10]))
    abstract = " ".join(r.abstracts())
    if abstract:
        if len(abstract) > max_abstract_chars:
            abstract = abstract[:max_abstract_chars].rsplit(" ", 1)[0] + " …"
        parts.append("Abstract: " + abstract)
    return " | ".join(parts)


def build_messages(query: str, condition: str, context: list[Record], max_abstract_chars: int = 600) -> tuple[str, str]:
    system, uses_retrieval = CONDITIONS[condition]
    if uses_retrieval:
        block = "\n".join(format_record(i, r, max_abstract_chars) for i, r in enumerate(context, start=1))
        header = "Catalogue records:" if system is GROUNDED_SYSTEM else "Possibly relevant catalogue records:"
        user = f"{header}\n{block or '(no records were retrieved)'}\n\nRequest: {query}"
    elif system is GROUNDED_SYSTEM:
        user = f"Catalogue records:\n(no records were retrieved)\n\nRequest: {query}"
    else:
        user = f"Request: {query}"
    return system, user


def abstain_message(lang: str | None) -> str:
    if lang == "ar":
        return "لا توجد في الفهرس سجلات ذات صلة بطلبك."
    return "I don't have relevant records in the catalogue for this request."
