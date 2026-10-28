"""Step 6 — retrieval experiments (RQ1, RQ2) with significance testing.

    python scripts/06_eval_retrieval.py

Tables written to results/tables/:
  retrieval_main.csv              every system, main template: means + 95% bootstrap CIs, zero-result rate
  retrieval_significance.csv      each system vs each baseline, paired randomisation test, Holm-adjusted
  retrieval_by_subgroup.csv       main metric by query language and query type
  retrieval_crosslingual.csv      recall of relevant records with / without a title in the query language
  retrieval_field_ablation.csv    metadata field templates (RQ2) for the ablation systems
  retrieval_language_ablation.csv parallel Arabic+English vs English-only vs Arabic-only metadata, by query language
  retrieval_field_completeness.csv  recall@10 of relevant records with/without abstract & subjects
  retrieval_latency.csv           seconds per query (query encoding + search)
  retrieval_qrels_sensitivity.csv main metric under alternative qrels (lenient, each annotator alone)
                                  and Kendall's tau between the resulting system rankings

Runs that did not build the judgement pool (systems outside `pool_systems`, MiniLM on
Arabic queries, and every ablation variant) leave some of their top documents unjudged.
Each table therefore also reports the main metric on condensed lists (`<metric>_cond`,
Sakai, 2007), where unjudged documents are removed before scoring; `pooled` says which
systems built the pool. Conclusions that hold under both versions are robust to pool bias.
"""
from __future__ import annotations

import pandas as pd

from _common import init, load_records, parser, path, results_path

from librag.eval.io import read_qrels, read_queries, write_run
from librag.eval.metrics import condense, per_query, zero_result_rate
from librag.eval.stats import bootstrap_ci, holm, paired_summary, wilson_ci
from librag.retrieval import SystemFactory, all_systems, language_variants

REPORT = ["ndcg@10", "ndcg@10_cond", "mrr@10", "p@5", "r@10", "r@100", "map", "success@1", "judged@10"]


UNPOOLED: set[str] = set()     # known-item queries judged only through their target (judged@k not meaningful)


def summarise(name: str, pq: pd.DataFrame, cfg: dict, extra: dict | None = None) -> dict:
    row = {"system": name, "n_queries": len(pq), **(extra or {})}
    for m in REPORT:
        if m not in pq:
            continue
        vals = pq.loc[~pq.index.isin(UNPOOLED), m] if m.startswith("judged@") else pq[m]
        if vals.empty:
            continue
        lo, hi = bootstrap_ci(vals.values, cfg["evaluation"]["n_bootstrap"], seed=cfg["project"]["seed"])
        row[m] = round(vals.mean(), 4)
        row[f"{m}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
    return row


def evaluate(run, qrels, cutoffs, main_metric: str) -> pd.DataFrame:
    """Per-query metrics plus the main metric on condensed lists."""
    pq = per_query(run, qrels, cutoffs)
    if not pq.empty:
        pqc = per_query(condense(run, qrels, UNPOOLED), qrels, cutoffs)
        pq[f"{main_metric}_cond"] = pqc[main_metric].reindex(pq.index).fillna(0.0)
    return pq


def main() -> None:
    cfg, _, log = init(parser(__doc__))
    e = cfg["evaluation"]
    seed, main_metric = cfg["project"]["seed"], e["main_metric"]
    records = load_records(cfg)
    queries = read_queries(path(cfg, "queries_file"))
    qrels = read_qrels(path(cfg, "qrels_file"))
    qmap = dict(zip(queries["qid"], queries["query"]))
    eval_qrels = {q: r for q, r in qrels.items() if q in qmap and any(v > 0 for v in r.values())}
    ki_file = path(cfg, "annotation_dir") / "known_item_qrels.csv"
    if ki_file.exists():
        UNPOOLED.update(pd.read_csv(ki_file, dtype={"qid": str}, encoding="utf-8-sig")["qid"])
        log.info(f"judged@k is computed over the {len(eval_qrels) - len(UNPOOLED & set(eval_qrels))} pooled queries "
                 f"(known-item queries are judged only through their target)")
    log.info(f"{len(eval_qrels)} answerable queries with judgements; {len(qmap)} queries in total")

    # ------------------------------------------------ main comparison
    factory = SystemFactory(cfg)
    per_q, rows = {}, []
    for system in all_systems(cfg):
        run = factory.run(system, qmap)
        write_run(run, results_path(cfg, "runs", f"{system.replace(':', '_').replace('+', '-')}.trec"), system)
        pq = evaluate(run, eval_qrels, e["cutoffs"], main_metric)
        pq.to_csv(results_path(cfg, "per_query", f"retrieval_{system.replace(':', '_').replace('+', '-')}.csv"))
        per_q[system] = pq
        excluded = [lang for lang, v in (e.get("pool_exclude") or {}).items() if system in (v or [])]
        pooled = "no" if system not in e["pool_systems"] else ("yes" if not excluded else f"not for {'/'.join(excluded)}")
        rows.append(summarise(system, pq, cfg, {"pooled": pooled,
                                                "zero_result_rate": round(zero_result_rate(run, list(qmap)), 4)}))
        log.info(f"{system:28s} {main_metric}={pq[main_metric].mean():.4f}  "
                 f"condensed={pq[f'{main_metric}_cond'].mean():.4f}  pooled={pooled}")
    pd.DataFrame(rows).to_csv(results_path(cfg, "tables", "retrieval_main.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ sensitivity to the relevance judgements (Voorhees, 2000)
    var_dir = path(cfg, "qrels_file").parent / "variants"
    if var_dir.exists():
        from scipy.stats import kendalltau

        base = {sname: per_q[sname][main_metric].mean() for sname in per_q}
        sens = [{"qrels": "main", "system": sname, main_metric: round(v, 4)} for sname, v in base.items()]
        taus = []
        for vf in sorted(var_dir.glob("*.txt")):
            vq = read_qrels(vf)
            vq = {q: vq.get(q, {}) for q in eval_qrels}            # same queries as the main evaluation
            scores = {}
            for sname in per_q:
                pqv = per_query(factory.run(sname, qmap), vq, e["cutoffs"])
                scores[sname] = pqv[main_metric].reindex(list(eval_qrels)).fillna(0).mean()
                sens.append({"qrels": vf.stem, "system": sname, main_metric: round(scores[sname], 4)})
            names = list(per_q)
            tau = kendalltau([base[n] for n in names], [scores[n] for n in names])[0]
            taus.append({"qrels": vf.stem, "system": "(Kendall tau vs main)", main_metric: round(float(tau), 3)})
            log.info(f"qrels variant {vf.stem:8s}: Kendall tau of system ranking vs main = {tau:.3f}")
        pd.DataFrame(sens + taus).to_csv(results_path(cfg, "tables", "retrieval_qrels_sensitivity.csv"),
                                         index=False, encoding="utf-8-sig")

    # ------------------------------------------------ significance vs baselines
    sig_rows = []
    for metric in [main_metric, "mrr@10", "r@10"]:
        pvals, tmp, seen = {}, [], set()
        for base in e["baselines"]:
            for system in per_q:
                if system == base or frozenset((system, base)) in seen:
                    continue
                seen.add(frozenset((system, base)))
                s = paired_summary(per_q[system][metric], per_q[base][metric],
                                   e["n_permutations"], e["n_bootstrap"], seed)
                key = f"{system} vs {base}"
                pvals[key] = s["p_value"]
                tmp.append({"metric": metric, "comparison": key, **{k: round(v, 4) if isinstance(v, float) else v
                                                                     for k, v in s.items()}})
        for a, b in e.get("extra_comparisons") or []:
            if a in per_q and b in per_q and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                s = paired_summary(per_q[a][metric], per_q[b][metric], e["n_permutations"], e["n_bootstrap"], seed)
                key = f"{a} vs {b}"
                pvals[key] = s["p_value"]
                tmp.append({"metric": metric, "comparison": key, **{k: round(v, 4) if isinstance(v, float) else v
                                                                     for k, v in s.items()}})
        adj = holm(pvals)
        for t in tmp:
            t["p_holm"] = round(adj[t["comparison"]], 4)
            t["significant"] = adj[t["comparison"]] < e["alpha"]
        sig_rows.extend(tmp)
    pd.DataFrame(sig_rows).to_csv(results_path(cfg, "tables", "retrieval_significance.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ subgroups: language and query type
    sub = []
    for system, pq in per_q.items():
        joined = pq.join(queries.set_index("qid")[["lang", "type"]])
        for col in ("lang", "type"):
            for val, grp in joined.groupby(col):
                lo, hi = bootstrap_ci(grp[main_metric].values, e["n_bootstrap"], seed=seed)
                sub.append({"system": system, "group": col, "value": val, "n": len(grp),
                            main_metric: round(grp[main_metric].mean(), 4), "ci": f"[{lo:.3f}, {hi:.3f}]"})
    pd.DataFrame(sub).to_csv(results_path(cfg, "tables", "retrieval_by_subgroup.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ cross-lingual recall
    # Is a relevant record found when it has NO title in the query's language?
    cl = []
    qlang = dict(zip(queries["qid"], queries["lang"]))
    for system in per_q:
        run = factory.run(system, qmap)
        counts = {True: [0, 0], False: [0, 0]}
        for qid, rels in eval_qrels.items():
            top = {d for d, _ in run.get(qid, [])[:10]}
            for d, g in rels.items():
                if g <= 0:
                    continue
                has_same = bool(records[d].title.get(qlang[qid]))
                counts[has_same][0] += 1
                counts[has_same][1] += int(d in top)
        for has_same, (n, h) in counts.items():
            lo, hi = wilson_ci(h, n)
            cl.append({"system": system,
                       "relevant_records": "title in query language" if has_same else "title only in other language",
                       "n": n, "recall@10": round(h / n, 4) if n else None, "ci": f"[{lo:.3f}, {hi:.3f}]"})
    pd.DataFrame(cl).to_csv(results_path(cfg, "tables", "retrieval_crosslingual.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ field and language ablation (RQ2)
    def run_variant(template: str, langs: str, system_short: str) -> pd.DataFrame:
        f = SystemFactory(cfg, template, langs)
        sys_name = "bm25" if system_short == "bm25" else f"dense:{system_short}"
        return evaluate(f.run(sys_name, qmap), eval_qrels, e["cutoffs"], main_metric)

    langs_main = cfg.get("template_languages", "all")
    abl, lang_abl = [], []
    for system_short in cfg["ablation_models"]:
        ref = run_variant(cfg["main_template"], langs_main, system_short)
        cond = f"{main_metric}_cond"
        start = len(abl)
        for t in cfg["ablation_templates"]:
            pq = run_variant(t, langs_main, system_short)
            s = paired_summary(pq[main_metric], ref[main_metric], e["n_permutations"], e["n_bootstrap"], seed)
            sc = paired_summary(pq[cond], ref[cond], e["n_permutations"], e["n_bootstrap"], seed)
            abl.append(summarise(system_short, pq, cfg, {"template": t, "fields": "+".join(cfg["templates"][t]),
                                                          f"diff_vs_{cfg['main_template']}": round(s["diff"], 4),
                                                          "p_value": round(s["p_value"], 4),
                                                          "diff_cond": round(sc["diff"], 4),
                                                          "p_value_cond": round(sc["p_value"], 4)}))
        for col in ("p_value", "p_value_cond"):                 # Holm within each system's template comparisons
            fam = {i: abl[i][col] for i in range(start, len(abl)) if abl[i]["template"] != cfg["main_template"]}
            for i, p_adj in holm(fam).items():
                abl[i][f"{col}_holm"] = round(p_adj, 4)
        qlang_s = queries.set_index("qid")["lang"]
        lstart = len(lang_abl)
        variants = {langs_main: ref, **{v: run_variant(cfg["main_template"], v, system_short)
                                         for v in language_variants(cfg)}}
        for variant, pq in variants.items():
            for qgroup in ("all", "ar", "en"):
                idx = pq.index if qgroup == "all" else pq.index[qlang_s.reindex(pq.index) == qgroup]
                if len(idx) == 0:
                    continue
                row = {"system": system_short, "metadata_languages": variant, "query_language": qgroup,
                       "n_queries": len(idx), main_metric: round(pq.loc[idx, main_metric].mean(), 4),
                       cond: round(pq.loc[idx, cond].mean(), 4)}
                if variant != langs_main:
                    s = paired_summary(pq.loc[idx, main_metric], ref.loc[idx, main_metric],
                                       e["n_permutations"], e["n_bootstrap"], seed)
                    sc = paired_summary(pq.loc[idx, cond], ref.loc[idx, cond], e["n_permutations"], e["n_bootstrap"], seed)
                    row.update({f"diff_vs_{langs_main}": round(s["diff"], 4),
                                "diff_ci": f"[{s['diff_ci_low']:.3f}, {s['diff_ci_high']:.3f}]",
                                "p_value": round(s["p_value"], 4),
                                "diff_cond": round(sc["diff"], 4), "p_value_cond": round(sc["p_value"], 4)})
                lang_abl.append(row)
        for col in ("p_value", "p_value_cond"):                 # Holm within each system's language comparisons
            fam = {i: lang_abl[i][col] for i in range(lstart, len(lang_abl)) if col in lang_abl[i]}
            for i, p_adj in holm(fam).items():
                lang_abl[i][f"{col}_holm"] = round(p_adj, 4)
    pd.DataFrame(abl).to_csv(results_path(cfg, "tables", "retrieval_field_ablation.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(lang_abl).to_csv(results_path(cfg, "tables", "retrieval_language_ablation.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ does catalogue completeness matter?
    comp = []
    for system in [s for s in per_q if s.startswith("hybrid") or s == "bm25"]:
        run = factory.run(system, qmap)
        stats: dict[tuple[str, bool], list[int]] = {}
        for qid, rels in eval_qrels.items():
            top = {d for d, _ in run.get(qid, [])[:10]}
            for d, g in rels.items():
                if g > 0:
                    for fld in ("abstract", "subjects"):
                        stats.setdefault((fld, records[d].has(fld)), []).append(int(d in top))
        for (fld, has), hits in sorted(stats.items()):
            lo, hi = wilson_ci(sum(hits), len(hits))
            comp.append({"system": system, "field": fld, "record_has_field": has, "relevant_records": len(hits),
                         "recall@10": round(sum(hits) / len(hits), 4), "ci": f"[{lo:.3f}, {hi:.3f}]"})
    pd.DataFrame(comp).to_csv(results_path(cfg, "tables", "retrieval_field_completeness.csv"), index=False, encoding="utf-8-sig")

    # ------------------------------------------------ single-query latency (as a user would experience it)
    import time

    import numpy as np

    lat = []
    sample = list(qmap.values())[:100]
    for system in per_q:
        factory.search(system, sample[0], 10)                      # warm-up (model load excluded)
        times = []
        for q in sample:
            factory._cache.clear()                                  # no cached component runs
            t0 = time.perf_counter()
            factory.search(system, q, 10)
            times.append(time.perf_counter() - t0)
        lat.append({"system": system, "n": len(times), "median_ms": round(1000 * float(np.median(times)), 2),
                    "p95_ms": round(1000 * float(np.percentile(times, 95)), 2)})
    pd.DataFrame(lat).to_csv(results_path(cfg, "tables", "retrieval_latency.csv"), index=False, encoding="utf-8-sig")
    log.info(f"tables → {results_path(cfg, 'tables')}")


if __name__ == "__main__":
    main()
