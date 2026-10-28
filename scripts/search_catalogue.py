"""Plain keyword search of the catalogue, for the human evaluation (step 8 workbooks).

    python scripts/search_catalogue.py

Type words from a title (Arabic or English) and press Enter; the ten best BM25 matches
are listed with their journal and year. Empty line to quit. The evaluator decides whether
the item in the workbook is one of these records; no similarity score is shown, so the
judgement does not lean on the automatic detector.

بحث بسيط في الفهرس للتقييم البشري: اكتب كلمات من العنوان ثم Enter.
"""
from __future__ import annotations

from _common import init, load_records, parser

from librag.retrieval import SystemFactory


def main() -> None:
    cfg, _, _ = init(parser(__doc__))
    records = load_records(cfg)
    factory = SystemFactory(cfg)
    print(f"{len(records)} records. Type title words (empty line to quit) / اكتب كلمات من العنوان")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        for n, (doc, _) in enumerate(factory.search("bm25", q, 10), start=1):
            r = records[doc]
            print(f"{n:2d}. {' / '.join(r.titles())}")
            print(f"    {'; '.join(r.authors[:4]) or '—'} | {r.year or '—'} | {r.collection or '—'}")


if __name__ == "__main__":
    main()
