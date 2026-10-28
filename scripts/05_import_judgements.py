"""Step 5 — read annotator workbooks, compute agreement, write qrels.

    python scripts/05_import_judgements.py
    python scripts/05_import_judgements.py --partial   # agreement so far, during annotation

Agreement (Cohen's κ binary and linearly weighted, Fleiss' κ for ≥3 annotators)
→ results/tables/agreement.csv. Disagreements are exported to
data/annotation/adjudication_needed.xlsx (with the query and the record). The
two annotators resolve them together, fill the column 'final' and save the file
as data/annotation/adjudication.xlsx, then this script is re-run. Rows left
empty, and all disagreements before that step, keep the lower grade (conservative).

When `evaluation.double_judged_fraction` < 1, the second annotator judges only a
sample of the queries: agreement and adjudication use those queries, and the
first annotator's grades are used for the others.

--partial computes agreement on the pairs both annotators have already judged
(a drift check part-way through the main annotation). It writes no qrels.

Besides the main qrels, alternative qrels are written to data/qrels/variants/ for a
sensitivity analysis in step 6 (Voorhees, 2000): `lenient` (the higher grade where the
annotators disagree) and, when every pair was judged by both, `A1` and `A2` alone.

--apply-labels sets the column `answerable` of the queries file from the judgements
(answerable = at least one record graded relevant in the main qrels). The labels the
writers gave are kept in a new column `answerable_written`. Run it once the qrels are final.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from _common import init, parser, path, results_path

from librag.eval.agreement import cohen_kappa, combine_judgements, fleiss_kappa, interpret_kappa, percent_agreement
from librag.eval.io import read_queries, write_qrels


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--partial", action="store_true", help="agreement on the pairs judged so far; no qrels")
    ap.add_argument("--apply-labels", action="store_true",
                    help="set 'answerable' in the queries file from the judgements (writers' labels kept)")
    cfg, args, log = init(ap)
    e = cfg["evaluation"]
    ann_dir = path(cfg, "annotation_dir")
    frames, context = {}, None
    for ann in e["annotators"]:
        p = ann_dir / f"judgements_{ann}.xlsx"
        df = pd.read_excel(p, sheet_name="judgements", dtype={"qid": str, "doc_id": str})
        missing = int(df["judgement"].isna().sum())
        if missing and not args.partial:
            raise SystemExit(f"{p}: {missing} pairs are not judged yet (use --partial for an interim agreement check)")
        if missing:
            log.info(f"{ann}: {len(df) - missing}/{len(df)} pairs judged so far")
            df = df.dropna(subset=["judgement"])
        df["judgement"] = pd.to_numeric(df["judgement"], errors="coerce")
        bad = ~df["judgement"].isin(e["relevance_scale"])
        if bad.any():
            raise SystemExit(f"{p}: invalid grades in rows {list(df.index[bad] + 2)[:10]}")
        frames[ann] = df.set_index(["qid", "doc_id"])["judgement"].astype(int)
        cols = [c for c in ("qid", "doc_id", "query", "target", "title", "subjects", "abstract") if c in df.columns]
        notes = df.set_index(["qid", "doc_id"])["note"].rename(f"note_{ann}") if "note" in df.columns else None
        ctx = df[cols].set_index(["qid", "doc_id"])
        context = ctx if context is None else context.combine_first(ctx)
        if notes is not None:
            context = context.join(notes, how="left")
    J, final = combine_judgements(frames, e["annotators"][0])
    if J.empty:
        raise SystemExit("no pair has been judged by every annotator yet")

    rows = []
    for a, b in combinations(J.columns, 2):
        rows.append({"pair": f"{a}-{b}",
                     "kappa_binary": cohen_kappa(J[a] > 0, J[b] > 0),
                     "kappa_weighted_linear": cohen_kappa(J[a], J[b], weights="linear"),
                     "percent_agreement": percent_agreement(J[a], J[b]),
                     "n_pairs": len(J)})
    if J.shape[1] >= 3:
        counts = np.stack([(J == g).sum(axis=1).values for g in e["relevance_scale"]], axis=1)
        rows.append({"pair": "all (Fleiss)", "kappa_weighted_linear": fleiss_kappa(counts), "n_pairs": len(J)})
    agree = pd.DataFrame(rows)
    agree["interpretation"] = agree["kappa_weighted_linear"].map(interpret_kappa)
    if args.partial:
        log.info(f"interim agreement on {len(J)} pairs judged by both annotators "
                 f"({J.index.get_level_values(0).nunique()} queries):\n" + agree.to_string(index=False))
        return
    agree.to_csv(results_path(cfg, "tables", "agreement.csv"), index=False, encoding="utf-8-sig")
    log.info("\n" + agree.to_string(index=False))
    single = len(final) - len(J)
    if single:
        log.info(f"{len(J)} pairs judged by every annotator; {single} pairs judged by {e['annotators'][0]} only")

    disagree = J[J.nunique(axis=1) > 1]
    lenient = final.copy()
    lenient.loc[J.index] = J.max(axis=1)
    adj_file = ann_dir / "adjudication.xlsx"
    if adj_file.exists():
        adj = pd.read_excel(adj_file, dtype={"qid": str, "doc_id": str})
        empty = int(adj["final"].isna().sum())
        adj = adj.dropna(subset=["final"])
        for _, row in adj.iterrows():
            final.loc[(row["qid"], row["doc_id"])] = int(row["final"])
            lenient.loc[(row["qid"], row["doc_id"])] = int(row["final"])
        log.info(f"applied {len(adj)} consensus grades; {empty} rows without consensus keep the lower grade")
    elif len(disagree):
        out = disagree.join(context, how="left").reset_index()
        out["final"] = ""
        lead = ["qid", "query", "title", "subjects", *e["annotators"], "final"]
        out = out[[c for c in lead if c in out.columns] + [c for c in out.columns if c not in lead]]
        out.to_excel(ann_dir / "adjudication_needed.xlsx", index=False)
        log.info(f"{len(disagree)} disagreements → {ann_dir / 'adjudication_needed.xlsx'} (using the lower grade for now)")

    ki_file = ann_dir / "known_item_qrels.csv"
    ki = pd.read_csv(ki_file, dtype={"qid": str, "doc_id": str}, encoding="utf-8-sig") if ki_file.exists() else None

    def to_qrels(grades: pd.Series) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for (qid, doc), g in grades.items():
            out.setdefault(str(qid), {})[str(doc)] = int(g)
        if ki is not None:
            for row in ki.itertuples():
                if row.qid not in out:                     # judged known-item queries keep their judgements
                    out[row.qid] = {row.doc_id: int(row.grade)}
        return out

    qrels = to_qrels(final)
    if ki is not None:
        log.info(f"known-item queries: {sum(q not in final.index.get_level_values(0) for q in ki['qid'])} targets "
                 f"added from {ki_file.name} (grade 2, R5-R6)")
    write_qrels(qrels, path(cfg, "qrels_file"))

    variants = {"lenient": lenient}
    if all(len(frames[a]) == len(J) for a in J.columns):          # every pair judged by every annotator
        variants.update({a: J[a] for a in J.columns})
    var_dir = path(cfg, "qrels_file").parent / "variants"
    var_dir.mkdir(parents=True, exist_ok=True)
    for name, grades in variants.items():
        write_qrels(to_qrels(grades), var_dir / f"{name}.txt")
    log.info(f"alternative qrels for the sensitivity analysis → {var_dir} ({', '.join(variants)})")

    queries = read_queries(path(cfg, "queries_file"))
    summary = []
    for qid, row in queries.iterrows():
        rels = qrels.get(qid, {})
        n_rel = sum(1 for v in rels.values() if v > 0)
        summary.append({"qid": qid, "lang": row["lang"], "type": row["type"], "answerable_label": row["answerable"],
                        "judged": len(rels), "relevant": n_rel, "highly_relevant": sum(1 for v in rels.values() if v == 2),
                        "answerable_judged": n_rel > 0,
                        "label_conflict": bool(row["answerable"]) != (n_rel > 0)})
    s = pd.DataFrame(summary)
    s.to_csv(results_path(cfg, "tables", "qrels_summary.csv"), index=False, encoding="utf-8-sig")
    conflicts = s[s["label_conflict"]]
    if args.apply_labels:
        apply_labels(path(cfg, "queries_file"), dict(zip(s["qid"], s["answerable_judged"])), log)
    elif len(conflicts):
        log.info(f"WARNING: {len(conflicts)} queries whose 'answerable' label disagrees with the judgements: "
                 f"{list(conflicts['qid'])}. Once the qrels are final, run again with --apply-labels.")
    log.info(f"qrels → {path(cfg, 'qrels_file')}  ({len(qrels)} queries, mean relevant per answerable query "
             f"{s.loc[s['relevant'] > 0, 'relevant'].mean():.1f})")


def apply_labels(queries_file, judged: dict, log) -> None:
    """Write judged answerability into the queries workbook, keeping the writers' labels."""
    from openpyxl import load_workbook

    wb = load_workbook(queries_file)
    ws = wb["queries"] if "queries" in wb.sheetnames else wb.worksheets[0]
    header = [str(c.value).strip().lower() if c.value is not None else "" for c in ws[1]]
    q_col, a_col = header.index("qid") + 1, header.index("answerable") + 1
    if "answerable_written" in header:
        w_col = header.index("answerable_written") + 1
    else:
        w_col = ws.max_column + 1
        ws.cell(1, w_col, "answerable_written")
        for r in range(2, ws.max_row + 1):
            ws.cell(r, w_col, ws.cell(r, a_col).value)
    changed = []
    for r in range(2, ws.max_row + 1):
        qid = str(ws.cell(r, q_col).value or "").strip()
        if qid in judged:
            new = 1 if judged[qid] else 0
            old = ws.cell(r, a_col).value
            if str(old).strip() != str(new):
                changed.append(qid)
            ws.cell(r, a_col, new)
    try:
        wb.save(queries_file)
    except PermissionError:
        raise SystemExit(f"cannot write {queries_file}: close it in Excel and run again")
    log.info(f"answerable labels set from the judgements ({len(changed)} changed: {changed}); "
             f"the writers' labels are kept in column 'answerable_written'")


if __name__ == "__main__":
    main()
