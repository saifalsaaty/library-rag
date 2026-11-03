"""Step 9 — human validation of the automatic hallucination labels and answer faithfulness.

    python scripts/09_import_human_eval.py

Reads results/generation/human_eval_<annotator>.xlsx (filled by the annotators) and writes:
  human_agreement.csv       Cohen's κ between annotators per question
  detector_validation.csv   precision / recall / κ of the automatic 'ungrounded' label vs humans (strict and
                            narrow definitions), and what humans found for 'catalog_not_in_context' items
  fabrication_breakdown.csv items not in the catalogue: truly fabricated vs real-but-not-held
  faithfulness.csv          share of unfaithful descriptions by model × condition
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from _common import init, parser, results_path

from librag.eval.agreement import cohen_kappa, percent_agreement
from librag.eval.stats import wilson_ci

COLS = {"in_provided_records(y/n)": "in_ctx", "in_catalogue(y/n)": "in_cat",
        "exists_elsewhere(y/n/unknown)": "exists", "author_correct(y/n/na)": "author_ok",
        "description_faithful(y/partial/n)": "faithful"}


def norm(v) -> str:
    s = str(v).strip().lower()
    return {"yes": "y", "نعم": "y", "no": "n", "لا": "n", "p": "partial", "جزئي": "partial"}.get(s, s)


def main() -> None:
    cfg, _, log = init(parser(__doc__))
    ann_names = cfg["evaluation"]["annotators"]
    keys = pd.read_csv(results_path(cfg, "generation", "human_eval_key.csv"), dtype=str).set_index("response_id")["key"]
    frames = {}
    for ann in ann_names:
        df = pd.read_excel(results_path(cfg, "generation", f"human_eval_{ann}.xlsx"), dtype=str).rename(columns=COLS)
        for c in COLS.values():
            df[c] = df[c].map(norm)
        frames[ann] = df.set_index(["response_id", "item_no"])
    ref = frames[ann_names[0]]

    # agreement
    agr = []
    for a, b in combinations(ann_names, 2):
        A, B = frames[a], frames[b]
        for c in ("in_ctx", "in_cat", "author_ok", "faithful"):
            both = pd.concat([A[c], B[c]], axis=1, keys=["a", "b"]).dropna()
            both = both[(both["a"] != "nan") & (both["b"] != "nan")]
            if len(both):
                agr.append({"pair": f"{a}-{b}", "question": c, "n": len(both),
                            "kappa": round(cohen_kappa(both["a"], both["b"]), 3),
                            "agreement": round(percent_agreement(both["a"], both["b"]), 3)})
    pd.DataFrame(agr).to_csv(results_path(cfg, "tables", "human_agreement.csv"), index=False, encoding="utf-8-sig")

    # automatic labels for the same items
    verified = results_path(cfg, "generation", "responses_verified.jsonl")      # written by step 8
    resp = pd.read_json(verified if verified.exists() else results_path(cfg, "generation", "responses.jsonl"),
                        lines=True, dtype={"qid": str})
    resp["key"] = resp["model"] + "|" + resp["condition"] + "|" + resp["k"].astype(str) + "|" + resp["qid"]
    auto = {}
    for _, row in resp.iterrows():
        for j, it in enumerate(row["verification"]["items"], start=1):
            auto[(row["key"], j)] = (it["label"], row["condition"])

    rows = []
    for (rid, item_no), h in ref.iterrows():
        key = keys.get(rid)
        lab = auto.get((key, int(item_no)))
        if key is None or lab is None:
            continue
        label, cond = lab
        rag = cond.startswith("rag")
        # strict definition (guard analysis): with retrieval, anything outside the supplied records is ungrounded
        human_ungrounded = (h["in_ctx"] == "n" if rag else h["in_cat"] == "n") or h["author_ok"] == "n"
        auto_ungrounded = label in ("not_in_catalog", "misattributed") or (rag and label == "catalog_not_in_context")
        # narrow definition (main generation table): the cited work is not in the catalog, or its authors are wrong
        human_narrow = h["in_cat"] == "n" or h["author_ok"] == "n"
        auto_narrow = label in ("not_in_catalog", "misattributed")
        model = key.split("|")[0]
        rows.append({"model": model, "condition": cond, "label": label, "human": human_ungrounded, "auto": auto_ungrounded,
                     "human_narrow": human_narrow, "auto_narrow": auto_narrow, "rag": rag,
                     "in_ctx": h["in_ctx"], "in_cat": h["in_cat"], "exists": h["exists"], "faithful": h["faithful"]})
    d = pd.DataFrame(rows)

    tp = int((d["human"] & d["auto"]).sum())
    fp = int((~d["human"] & d["auto"]).sum())
    fn = int((d["human"] & ~d["auto"]).sum())
    tn = int((~d["human"] & ~d["auto"]).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    # items were sampled per automatic label: weight each by (items with that label) / (items sampled)
    design_file = results_path(cfg, "generation", "human_eval_design.csv")
    w = pd.Series(1.0, index=d.index)
    if design_file.exists() and len(d):
        des = pd.read_csv(design_file).set_index("label")
        w = d["label"].map(lambda lab: des.loc[lab, "population_items"] / des.loc[lab, "sampled_items"]
                           if lab in des.index and des.loc[lab, "sampled_items"] else 1.0)
    wtp, wfp = float(w[d["human"] & d["auto"]].sum()), float(w[~d["human"] & d["auto"]].sum())
    wfn = float(w[d["human"] & ~d["auto"]].sum())
    wprec = wtp / (wtp + wfp) if wtp + wfp else float("nan")
    wrec = wtp / (wtp + wfn) if wtp + wfn else float("nan")
    by_label = {lab: round(float((g["human"] == g["auto"]).mean()), 3) for lab, g in d.groupby("label")} if len(d) else {}

    def pr(h, a):
        """unweighted and design-weighted precision / recall of the automatic label against the human one"""
        t, f_p, f_n = (h & a), (~h & a), (h & ~a)
        out = [t.sum() / (t.sum() + f_p.sum()), t.sum() / (t.sum() + f_n.sum()),
               w[t].sum() / (w[t].sum() + w[f_p].sum()), w[t].sum() / (w[t].sum() + w[f_n].sum())]
        return [round(float(x), 3) if x == x else None for x in out]

    n_p, n_r, n_wp, n_wr = pr(d["human_narrow"], d["auto_narrow"]) if len(d) else [None] * 4
    cnic = d[d["label"] == "catalog_not_in_context"] if len(d) else d
    pd.DataFrame([{"items": len(d), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                   "precision": round(prec, 3), "recall": round(rec, 3),
                   "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else None,
                   "accuracy": round((tp + tn) / len(d), 3) if len(d) else None,
                   "kappa_vs_human": round(cohen_kappa(d["human"], d["auto"]), 3) if len(d) else None,
                   "precision_weighted": round(wprec, 3), "recall_weighted": round(wrec, 3),
                   "narrow_precision": n_p, "narrow_recall": n_r,
                   "narrow_precision_weighted": n_wp, "narrow_recall_weighted": n_wr,
                   "cnic_items": len(cnic), "cnic_human_not_in_catalog": int((cnic["in_cat"] == "n").sum()),
                   "cnic_from_no_retrieval": int((~cnic["rag"].astype(bool)).sum()),
                   "agreement_by_auto_label": str(by_label)}]) \
        .to_csv(results_path(cfg, "tables", "detector_validation.csv"), index=False, encoding="utf-8-sig")

    notcat = d[d["in_cat"] == "n"]
    fab = []
    for (model, cond), grp in notcat.groupby(["model", "condition"]):
        n = len(grp)
        k = int((grp["exists"] == "n").sum())
        lo, hi = wilson_ci(k, n)
        fab.append({"model": model, "condition": cond, "items_not_in_catalogue": n,
                    "fabricated (exists nowhere)": k, "real_but_not_held": int((grp["exists"] == "y").sum()),
                    "unknown": int((grp["exists"] == "unknown").sum()),
                    "fabricated_share": round(k / n, 3) if n else None, "ci": f"[{lo:.3f}, {hi:.3f}]"})
    pd.DataFrame(fab).to_csv(results_path(cfg, "tables", "fabrication_breakdown.csv"), index=False, encoding="utf-8-sig")

    faith = []
    for (model, cond), grp in d[d["faithful"].isin(["y", "partial", "n"])].groupby(["model", "condition"]):
        n = len(grp)
        k = int((grp["faithful"] == "n").sum())
        lo, hi = wilson_ci(k, n)
        faith.append({"model": model, "condition": cond, "n_items": n,
                      "faithful_%": round(100 * (grp["faithful"] == "y").mean(), 1),
                      "partial_%": round(100 * (grp["faithful"] == "partial").mean(), 1),
                      "unfaithful_%": round(100 * k / n, 1), "unfaithful_ci": f"[{lo:.3f}, {hi:.3f}]"})
    pd.DataFrame(faith).to_csv(results_path(cfg, "tables", "faithfulness.csv"), index=False, encoding="utf-8-sig")
    log.info(f"detector vs human: precision={prec:.3f} recall={rec:.3f} on {len(d)} items")


if __name__ == "__main__":
    main()
