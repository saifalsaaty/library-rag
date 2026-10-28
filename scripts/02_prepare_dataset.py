"""Step 2 — clean, de-duplicate and describe the collection (Data section, Table 1–2).

    python scripts/02_prepare_dataset.py

Outputs
  data/processed/records.jsonl          the evaluation collection
  results/tables/dataset_stats.json     all descriptive statistics
  results/tables/dataset_overview.csv   one-row summary per source/collection
"""
from __future__ import annotations

import pandas as pd

from _common import init, parser, path, results_path

from librag.config import write_json
from librag.dataset import clean_record, deduplicate, describe, stratified_sample
from librag.schema import read_jsonl, write_jsonl
from librag.text import normalize


def main() -> None:
    cfg, _, log = init(parser(__doc__))
    d = cfg["dataset"]
    placeholders = {normalize(p) for p in d["placeholder_values"]}
    raw_dir = path(cfg, "raw_dir")

    loaded, kept = 0, []
    for src in d["sources"]:
        p = raw_dir / f"{src['name']}.jsonl"
        if not p.exists():
            log.info(f"skip {src['name']}: {p} not found (run 01_collect_records.py)")
            continue
        for r in read_jsonl(p):
            loaded += 1
            r = clean_record(r, placeholders, d["min_title_chars"])
            if r is not None:
                kept.append(r)
    if not kept:
        raise SystemExit("no records found — run scripts/01_collect_records.py first")

    unique, n_dupes = deduplicate(kept)
    if d.get("max_records") and len(unique) > d["max_records"]:
        unique = stratified_sample(unique, d["max_records"], cfg["project"]["seed"])
    unique.sort(key=lambda r: r.id)

    out = path(cfg, "processed_dir") / "records.jsonl"
    write_jsonl(unique, out)

    stats = describe(unique)
    stats["pipeline"] = {"loaded": loaded, "dropped_empty_or_placeholder": loaded - len(kept),
                         "duplicates_removed": n_dupes, "final": len(unique)}
    write_json(stats, results_path(cfg, "tables", "dataset_stats.json"))

    rows = []
    for (src, col), grp in pd.DataFrame(
            [{"source": r.source, "collection": r.collection or "-", "rec": r} for r in unique]
    ).groupby(["source", "collection"]):
        s = describe(list(grp["rec"]))
        rows.append({"source": src, "collection": col, "records": s["n_records"],
                     **{f"has_{k}_%": v for k, v in s["field_completeness_pct"].items()},
                     "bilingual_%": s["bilingual_ar_en_pct"],
                     "arabic_title_%": s["has_arabic_title_pct"], "english_title_%": s["has_english_title_pct"],
                     "years": f"{s['year_min']}–{s['year_max']}"})
    pd.DataFrame(rows).to_csv(results_path(cfg, "tables", "dataset_overview.csv"), index=False, encoding="utf-8-sig")

    log.info(f"loaded {loaded} → kept {len(kept)} → unique {len(unique)} (duplicates removed: {n_dupes})")
    log.info(f"collections: {len(stats['collections'])}  {stats['collections']}")
    log.info(f"title languages: {stats['title_languages']}")
    log.info(f"field completeness %: {stats['field_completeness_pct']}")
    log.info(f"bilingual Arabic–English records: {stats['bilingual_ar_en_pct']}%")


if __name__ == "__main__":
    main()
