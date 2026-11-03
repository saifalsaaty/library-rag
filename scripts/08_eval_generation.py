"""Step 8 — hallucination, abstention, usefulness and latency of the generation runs (RQ3, RQ4).

    python scripts/08_eval_generation.py

Tables written to results/tables/:
  generation_main.csv          model × condition at the main k (Wilson 95% CIs)
  generation_guard.csv         effect of the verification guard on rag conditions
  generation_mcnemar.csv       paired tests between conditions (Holm-adjusted)
  generation_k_ablation.csv    number of retrieved records k
  generation_by_language.csv   Arabic vs English queries
  generation_threshold_sensitivity.csv   robustness of the automatic labels to the matching threshold
  generation_abstention_by_context.csv   abstention when the retrieved records do / do not include a
                                         record judged relevant (separates retrieval misses from model errors)
and results/generation/human_eval_<annotator>.xlsx for manual validation.

A response cut off by the token limit (done_reason = "length") ends with an incomplete
line; that last item is not scored. The share of truncated responses is reported.

Every response is verified again here with the current detector (and the guarded answer
rebuilt from that verification), so all tables use one version of the detector even if it
was improved after generation. The result is saved as results/generation/responses_verified.jsonl
(read by step 9) together with the number of responses whose labels changed.
"""
from __future__ import annotations

import hashlib
import random

import numpy as np
import pandas as pd

from _common import init, load_records, parser, path, results_path

from librag.eval.io import read_qrels
from librag.eval.stats import holm, mcnemar_exact, wilson_ci
from librag.generation.verify import CatalogMatcher, guard, verify

UNGROUNDED = ("not_in_catalog", "misattributed")


def rate(series) -> tuple[float, str]:
    s = pd.Series(series).astype(bool)
    if len(s) == 0:
        return float("nan"), ""
    lo, hi = wilson_ci(int(s.sum()), len(s))
    return round(float(s.mean()), 4), f"[{lo:.3f}, {hi:.3f}]"


def scored_items(verification: dict, llm: dict) -> list[dict]:
    """Items that count: the last line of a response cut off by the token limit is incomplete and dropped."""
    items = list(verification.get("items") or [])
    if llm.get("done_reason") == "length" and items:
        items = items[:-1]
    return items


def flatten(df: pd.DataFrame, qrels: dict) -> pd.DataFrame:
    v = df["verification"]
    items = [scored_items(vv, ll) for vv, ll in zip(df["verification"], df["llm"])]
    count = lambda lab: [sum(1 for i in it if i["label"] == lab) for it in items]  # noqa: E731
    df = df.assign(
        abstained=v.map(lambda x: x["abstained"]),
        truncated=df["llm"].map(lambda x: x.get("done_reason") == "length"),
        n_items=[len(it) for it in items],
        n_grounded=count("grounded"),
        n_not_in_catalog=count("not_in_catalog"),
        n_misattributed=count("misattributed"),
        n_cat_not_ctx=count("catalog_not_in_context"),
        total_s=df["llm"].map(lambda x: x["total_s"] - x["load_s"]),
        tokens_per_s=df["llm"].map(lambda x: x["tokens_per_s"]),
        output_tokens=df["llm"].map(lambda x: x["output_tokens"]),
    )
    df["hallucinated"] = (df["n_not_in_catalog"] + df["n_misattributed"]) > 0
    df["ungrounded_any"] = df["hallucinated"] | ((df["n_cat_not_ctx"] > 0) & df["condition"].str.startswith("rag"))

    def relevance(row):
        rels = qrels.get(row["qid"], {})
        ids = [i["record_id"] for i in scored_items(row["verification"], row["llm"])
               if i["label"] == "grounded" and i["record_id"]]
        judged = [rels[i] for i in ids if i in rels]
        return pd.Series({"rec_items": len(ids), "rec_judged": len(judged), "rec_relevant": sum(1 for g in judged if g > 0)})

    return pd.concat([df, df.apply(relevance, axis=1)], axis=1)


def main() -> None:
    cfg, _, log = init(parser(__doc__))
    g, e = cfg["generation"], cfg["evaluation"]
    records = load_records(cfg)
    qrels = read_qrels(path(cfg, "qrels_file")) if path(cfg, "qrels_file").exists() else {}
    raw = pd.read_json(results_path(cfg, "generation", "responses.jsonl"), lines=True, dtype={"qid": str})
    matcher = CatalogMatcher(records, g["verification"]["candidate_pool"])
    thr = g["verification"]["title_threshold"]
    new_v, new_guard, changed = [], [], 0
    for _, row in raw.iterrows():
        v = verify(row["response"], row["context_ids"], matcher, thr)
        vd = v.to_dict()
        old = row["verification"]
        if vd["abstained"] != old["abstained"] or [i["label"] for i in vd["items"]] != [i["label"] for i in old["items"]]:
            changed += 1
        new_v.append(vd)
        new_guard.append(guard(v, row["context_ids"], records, row["lang"]) if row["condition"].startswith("rag") else None)
    raw["verification_at_generation"] = raw["verification"]
    raw["verification"], raw["guarded_response"] = new_v, new_guard
    raw.to_json(results_path(cfg, "generation", "responses_verified.jsonl"), orient="records", lines=True,
                force_ascii=False)
    log.info(f"re-verified {len(raw)} responses with the current detector; labels changed in {changed}")
    df = flatten(raw, qrels)
    main_k = g["k"]
    main = df[(df["k"] == main_k) | (df["k"] == 0)]

    # ------------------------------------------------ main table
    rows = []
    for (model, cond), grp in main.groupby(["model", "condition"], sort=False):
        ans, una = grp[grp["answerable"]], grp[~grp["answerable"]]
        answered = ans[~ans["abstained"]]
        h, h_ci = rate(answered["hallucinated"])
        fa, fa_ci = rate(ans["abstained"])
        ca, ca_ci = rate(una["abstained"])
        items = answered["n_items"].sum()
        rows.append({
            "model": model, "condition": cond, "n_answerable": len(ans), "n_unanswerable": len(una),
            "hallucination_rate": h, "hallucination_ci": h_ci,
            "not_in_catalog_items_%": round(100 * answered["n_not_in_catalog"].sum() / items, 2) if items else None,
            "misattributed_items_%": round(100 * answered["n_misattributed"].sum() / items, 2) if items else None,
            "citation_precision": round(answered["n_grounded"].sum() / items, 4) if items else None,
            "false_abstention_rate": fa, "false_abstention_ci": fa_ci,
            "correct_abstention_rate": ca, "correct_abstention_ci": ca_ci,
            "recommendation_precision": round(answered["rec_relevant"].sum() / answered["rec_judged"].sum(), 4)
            if answered["rec_judged"].sum() else None,
            "recommendation_judged_%": round(100 * answered["rec_judged"].sum() / answered["rec_items"].sum(), 1)
            if answered["rec_items"].sum() else None,
            "latency_median_s": round(grp["total_s"].median(), 3),
            "latency_p95_s": round(grp["total_s"].quantile(0.95), 3),
            "tokens_per_s_median": round(grp["tokens_per_s"].median(), 1),
            "truncated_%": round(100 * grp["truncated"].mean(), 1),
        })
    pd.DataFrame(rows).to_csv(results_path(cfg, "tables", "generation_main.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ abstention given the retrieved context
    ctx_rows = []
    for (model, cond), grp in main[main["condition"].str.startswith("rag")].groupby(["model", "condition"], sort=False):
        has_rel = grp.apply(lambda r: any(qrels.get(r["qid"], {}).get(d, 0) > 0 for d in r["context_ids"]), axis=1)
        a1, c1 = rate(grp.loc[has_rel, "abstained"])
        a0, c0 = rate(grp.loc[~has_rel, "abstained"])
        ctx_rows.append({"model": model, "condition": cond,
                         "n_relevant_in_context": int(has_rel.sum()), "abstention_rate_relevant_in_context": a1, "ci_1": c1,
                         "n_no_relevant_in_context": int((~has_rel).sum()), "abstention_rate_no_relevant_in_context": a0,
                         "ci_0": c0})
    pd.DataFrame(ctx_rows).to_csv(results_path(cfg, "tables", "generation_abstention_by_context.csv"),
                                  index=False, encoding="utf-8-sig")

    # ------------------------------------------------ verification guard (re-verified for honesty)
    guard_rows = []
    for (model, cond), grp in main[main["condition"].str.startswith("rag")].groupby(["model", "condition"], sort=False):
        ans = grp[grp["answerable"]]
        after_ungrounded, after_abstain, kept = [], [], []
        for _, row in ans.iterrows():
            gv = verify(row["guarded_response"], row["context_ids"], matcher, g["verification"]["title_threshold"])
            after_ungrounded.append(gv.counts()["not_in_catalog"] + gv.counts()["misattributed"] > 0)
            after_abstain.append(gv.abstained)
            kept.append(gv.counts()["grounded"])
        before, _ = rate(ans["ungrounded_any"])
        after, after_ci = rate(after_ungrounded)
        fa_b, _ = rate(ans["abstained"])
        fa_a, fa_a_ci = rate(after_abstain)
        guard_rows.append({"model": model, "condition": cond, "ungrounded_before": before, "ungrounded_after": after,
                           "ungrounded_after_ci": after_ci, "false_abstention_before": fa_b,
                           "false_abstention_after": fa_a, "false_abstention_after_ci": fa_a_ci,
                           "mean_items_kept": round(float(np.mean(kept)), 2) if kept else None})
    pd.DataFrame(guard_rows).to_csv(results_path(cfg, "tables", "generation_guard.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ paired tests between conditions (same model, same queries)
    contrasts = [("rag_grounded", "rag_plain", "prompt effect with retrieval"),
                 ("norag_grounded", "norag_plain", "prompt effect without retrieval"),
                 ("rag_plain", "norag_plain", "retrieval effect, plain prompt"),
                 ("rag_grounded", "norag_grounded", "retrieval effect, grounded prompt")]
    tests, pvals = [], {}
    ans_main = main[main["answerable"]]
    for model, grp in ans_main.groupby("model"):
        piv = grp.pivot_table(index="qid", columns="condition", values="hallucinated", aggfunc="first")
        for a, b, label in contrasts:
            if a in piv and b in piv:
                pair = piv[[a, b]].dropna()
                r = mcnemar_exact(pair[a].astype(bool), pair[b].astype(bool))
                key = f"{model}: {a} vs {b}"
                pvals[key] = r["p_value"]
                tests.append({"model": model, "contrast": label, "a": a, "b": b, "n": len(pair),
                              "rate_a": round(pair[a].astype(bool).mean(), 4), "rate_b": round(pair[b].astype(bool).mean(), 4),
                              **r, "key": key})
    adj = holm(pvals)
    for t in tests:
        t["p_holm"] = round(adj[t.pop("key")], 4)
    pd.DataFrame(tests).to_csv(results_path(cfg, "tables", "generation_mcnemar.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ k ablation
    kab = []
    for (model, k), grp in df[df["condition"] == "rag_grounded"].groupby(["model", "k"]):
        ans = grp[grp["answerable"]]
        answered = ans[~ans["abstained"]]
        h, h_ci = rate(answered["hallucinated"])
        fa, fa_ci = rate(ans["abstained"])
        kab.append({"model": model, "k": k, "n": len(ans), "hallucination_rate": h, "ci": h_ci,
                    "false_abstention_rate": fa, "fa_ci": fa_ci,
                    "recommendation_precision": round(answered["rec_relevant"].sum() / answered["rec_judged"].sum(), 4)
                    if answered["rec_judged"].sum() else None,
                    "prompt_tokens_median": float(grp["llm"].map(lambda x: x["prompt_tokens"]).median()),
                    "latency_median_s": round(grp["total_s"].median(), 3)})
    pd.DataFrame(kab).to_csv(results_path(cfg, "tables", "generation_k_ablation.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ Arabic vs English queries
    lang_rows = []
    for (model, cond, lang), grp in main[main["answerable"]].groupby(["model", "condition", "lang"]):
        answered = grp[~grp["abstained"]]
        h, h_ci = rate(answered["hallucinated"])
        fa, fa_ci = rate(grp["abstained"])
        lang_rows.append({"model": model, "condition": cond, "lang": lang, "n": len(grp),
                          "hallucination_rate": h, "ci": h_ci, "false_abstention_rate": fa, "fa_ci": fa_ci})
    pd.DataFrame(lang_rows).to_csv(results_path(cfg, "tables", "generation_by_language.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ threshold sensitivity of the automatic detector
    sens = []
    for thr in g["verification"]["sensitivity_thresholds"]:
        flags = []
        for _, row in main[main["answerable"]].iterrows():
            vv = verify(row["response"], row["context_ids"], matcher, thr)
            # same scoring rule as the main table: the incomplete last line of a truncated response is dropped
            kept = scored_items(vv.to_dict(), row["llm"])
            flags.append({"model": row["model"], "condition": row["condition"], "abstained": vv.abstained,
                          "hallucinated": any(i["label"] in ("not_in_catalog", "misattributed") for i in kept)})
        f = pd.DataFrame(flags)
        for (model, cond), grp in f.groupby(["model", "condition"]):
            h, ci = rate(grp.loc[~grp["abstained"], "hallucinated"])
            sens.append({"threshold": thr, "model": model, "condition": cond, "hallucination_rate": h, "ci": ci})
    pd.DataFrame(sens).to_csv(results_path(cfg, "tables", "generation_threshold_sensitivity.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ human evaluation workbooks
    # Items (not whole responses) are sampled, stratified by the automatic label, so that each label's
    # precision can be checked with a manageable workload. Step 9 re-weights by the population sizes
    # in human_eval_design.csv to estimate the detector's overall precision and recall.
    he = cfg["evaluation"].get("human_eval", {})
    quotas = he.get("items_per_label", {"not_in_catalog": 60, "grounded": 40,
                                         "misattributed": 30, "catalog_not_in_context": 20})
    rng = random.Random(cfg["project"]["seed"])
    pool: dict[str, list[tuple[int, int]]] = {}
    for i, row in main.iterrows():
        for j, it in enumerate(scored_items(row["verification"], row["llm"]), start=1):
            pool.setdefault(it["label"], []).append((i, j))
    picked, design = [], []
    for label, cands in sorted(pool.items()):
        take = rng.sample(cands, min(len(cands), int(quotas.get(label, 0))))
        picked += take
        design.append({"label": label, "population_items": len(cands), "sampled_items": len(take)})
    pd.DataFrame(design).to_csv(results_path(cfg, "generation", "human_eval_design.csv"), index=False, encoding="utf-8-sig")
    sample_rows, keys = [], {}
    for i, j in picked:
        row = df.loc[i]
        ctx = "\n".join(f"[{n}] {records[d].primary_title()} — {'; '.join(records[d].authors[:3])}"
                        for n, d in enumerate(row["context_ids"], start=1))
        key = f"{row['model']}|{row['condition']}|{row['k']}|{row['qid']}"
        rid = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        keys[rid] = key
        sample_rows.append({"response_id": rid, "item_no": j, "query": row["query"],
                            "provided_records": ctx or "(none)", "full_response": row["response"],
                            "item_line": row["verification"]["items"][j - 1].get("line", ""),
                            "in_provided_records(y/n)": "", "in_catalogue(y/n)": "",
                            "exists_elsewhere(y/n/unknown)": "", "author_correct(y/n/na)": "",
                            "description_faithful(y/partial/n)": "", "note": ""})
    sample = pd.DataFrame(sample_rows).sample(frac=1.0, random_state=cfg["project"]["seed"])  # hide label order
    pd.DataFrame([{"response_id": k, "key": v} for k, v in keys.items()]).to_csv(
        results_path(cfg, "generation", "human_eval_key.csv"), index=False, encoding="utf-8-sig")
    n_second = int(he.get("second_annotator_items", 50))
    for a, ann in enumerate(e["annotators"]):
        target = results_path(cfg, "generation", f"human_eval_{ann}.xlsx")
        part = sample if a == 0 else sample.sample(n=min(n_second, len(sample)), random_state=cfg["project"]["seed"] + a)
        if not target.exists():
            part.to_excel(target, index=False)
            log.info(f"human evaluation workbook → {target} ({len(part)} items)")

    log.info("\n" + pd.DataFrame(rows)[["model", "condition", "hallucination_rate", "false_abstention_rate",
                                       "correct_abstention_rate", "latency_median_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
