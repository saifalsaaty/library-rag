"""Step 1 — collect real bibliographic records from every source in config.yaml.

    python scripts/01_collect_records.py                                   # everything
    python scripts/01_collect_records.py --only uomosul_ojs                # one source
    python scripts/01_collect_records.py --only uomosul_ojs --journals rsci mmed
                                       # re-harvest some journals and merge them into the existing file

Raw OAI pages are kept under data/raw/oai/<source>/<journal>/ for provenance;
parsed records go to data/raw/<source>.jsonl. A per-journal summary is written
to results/tables/harvest_summary.csv.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from _common import init, parser, path, results_path

from librag.loaders import load_source
from librag.schema import read_jsonl, write_jsonl


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--only", help="collect a single source by name")
    ap.add_argument("--journals", nargs="*", help="(OJS sources) only these journal codes; merged into the existing file")
    cfg, args, log = init(ap)
    raw_dir = path(cfg, "raw_dir")
    for src in cfg["dataset"]["sources"]:
        if args.only and src["name"] != args.only:
            continue
        out = raw_dir / f"{src['name']}.jsonl"
        if args.journals:
            src = {**src, "journals": args.journals}
            new = list(load_source(src, raw_dir, log=log.info))
            keep = [r for r in read_jsonl(out) if r.collection not in set(args.journals)] if out.exists() else []
            n = write_jsonl(keep + new, out)
        else:
            n = write_jsonl(load_source(src, raw_dir, log=log.info), out)
        log.info(f"{src['name']}: {n} records written to {out}")

        counts = Counter(r.collection or "-" for r in read_jsonl(out))
        summary = pd.DataFrame(sorted(counts.items()), columns=["collection", "records"])
        summary.to_csv(results_path(cfg, "tables", f"harvest_summary_{src['name']}.csv"), index=False, encoding="utf-8-sig")
        log.info("records per collection:\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
