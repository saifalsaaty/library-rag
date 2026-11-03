"""Step 3 — build BM25 and dense indexes for the main configuration and the ablations.

    python scripts/03_build_indexes.py               # everything
    python scripts/03_build_indexes.py --bm25-only   # quick check without a GPU
    python scripts/03_build_indexes.py --only TS:bge_m3          # rebuild one index (overwrites it)
    python scripts/03_build_indexes.py --only full/en:bm25 T:bge_m3

Main configuration : main_template × {bm25 + every dense model}
Field ablation     : ablation_templates × ablation_models
Language ablation  : main_template with English-only / Arabic-only metadata × ablation_models
Build time, throughput and truncation statistics → results/tables/index_build.csv
"""
from __future__ import annotations

import time

import pandas as pd

from _common import init, load_records, parser, results_path

from librag.dataset import record_text
from librag.retrieval import dense_spec, language_variants, variant_dir
from librag.retrieval.bm25 import BM25
from librag.retrieval.dense import DenseEncoder, build_dense_index


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--bm25-only", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", nargs="*", metavar="TEMPLATE[/LANG]:SYSTEM",
                    help="rebuild only these indexes, e.g. TS:bge_m3 or full/en:bm25 (existing files are overwritten)")
    cfg, args, log = init(ap)
    records = load_records(cfg)
    ids = sorted(records)
    r = cfg["retrieval"]
    langs_main = cfg.get("template_languages", "all")
    dense_names = [m["name"] for m in r["dense_models"]]

    jobs: list[tuple[str, str, str]] = []                 # (template, languages, system)
    for sys_name in ["bm25"] + dense_names:
        jobs.append((cfg["main_template"], langs_main, sys_name))
    for t in cfg["ablation_templates"]:
        for sys_name in cfg["ablation_models"]:
            jobs.append((t, langs_main, sys_name))
    for variant in language_variants(cfg):
        for sys_name in cfg["ablation_models"]:
            jobs.append((cfg["main_template"], variant, sys_name))
    jobs = list(dict.fromkeys(jobs))
    if args.bm25_only:
        jobs = [j for j in jobs if j[2] == "bm25"]
    if args.only:
        wanted = set(args.only)
        jobs = [j for j in jobs if f"{j[0]}:{j[2]}" in wanted or f"{j[0]}/{j[1]}:{j[2]}" in wanted]
        if not jobs:
            raise SystemExit(f"no index matches {args.only}; use names like TS:bge_m3 or full/en:bm25")
        args.skip_existing = False

    stats, encoders = [], {}
    for template, langs, sys_name in jobs:
        out_dir = variant_dir(cfg, template, langs)
        texts = [record_text(records[i], cfg["templates"][template], langs) for i in ids]
        if sys_name == "bm25":
            target = out_dir / "bm25.pkl"
            if args.skip_existing and target.exists():
                continue
            t0 = time.perf_counter()
            bm = BM25(r["bm25"]["k1"], r["bm25"]["b"], r["bm25"]["stem"], r["bm25"]["remove_stopwords"]).fit(ids, texts)
            bm.save(target)
            stats.append({"template": template, "languages": langs, "system": "bm25",
                          "build_seconds": round(time.perf_counter() - t0, 2), "vocab": len(bm.vocab),
                          "n_docs": len(ids)})
        else:
            target = out_dir / "dense" / sys_name
            if args.skip_existing and (target / "embeddings.npy").exists():
                continue
            spec = dense_spec(cfg, sys_name)
            if sys_name not in encoders:
                encoders[sys_name] = DenseEncoder(spec["model_id"], r["device"], r["max_seq_length_cap"], r["batch_size"])
            meta = build_dense_index(encoders[sys_name], ids, texts, spec["doc_prefix"], target)
            stats.append({"template": template, "languages": langs, "system": f"dense:{sys_name}", **meta})
        log.info(f"built {sys_name} [{template}/{langs}]")

    if stats:
        out = results_path(cfg, "tables", "index_build.csv")
        old = pd.read_csv(out) if out.exists() else pd.DataFrame()
        new = pd.concat([old, pd.DataFrame(stats)]).drop_duplicates(["template", "languages", "system"], keep="last")
        new.to_csv(out, index=False, encoding="utf-8-sig")
        log.info(f"build statistics → {out}")


if __name__ == "__main__":
    main()
