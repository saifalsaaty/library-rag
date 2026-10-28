"""Step 11 — docs/DETAILS.md: the implementation details that the article summarises, built from this
project's own configuration, logs and result files (nothing is typed by hand).

    python scripts/11_make_details.py

Every number below is read from a file in this folder; the source file is named under each table.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from _common import init, parser

PARTIAL_NOTE = "server failed at a fixed point of the result list; the records before it were kept"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(rows)


def read_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def harvest_section(root: Path, cfg: dict) -> list[str]:
    src = cfg["dataset"]["sources"][0]
    journals = src["journals"]
    hs = read_csv(root / "results/tables/harvest_summary_uomosul_ojs.csv")
    got = dict(zip(hs["collection"], hs["records"])) if hs is not None else {}
    reasons = {}
    log = root / "results/logs/01_collect_records.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"!! journal '(\w+)'[: ]+(.*?)(?: — re-run later.*)?$", line)
            if m:
                j, msg = m.group(1), m.group(2)
                if "HTML page" in msg:
                    reasons[j] = "OAI-PMH endpoint returns an HTML page (OAI disabled or login required)"
                elif "page 1" in msg and j not in got:
                    reasons[j] = "HTTP 500 errors on the first page (two separate harvests)"
                elif j in got:
                    reasons[j] = PARTIAL_NOTE
    rows = []
    for j in journals:
        info = root / f"data/raw/oai/{src['name']}/{j}/HARVEST_INFO.txt"
        kv = dict(l.split("=", 1) for l in info.read_text(encoding="utf-8").splitlines() if "=" in l) if info.exists() else {}
        if j in got:
            status = "included (partial)" if reasons.get(j) == PARTIAL_NOTE else "included"
            note = reasons.get(j, "")
        else:
            status = "excluded"
            note = reasons.get(j) or ("only deleted-record placeholders" if kv.get("records_seen", "0") != "0"
                                      and kv.get("records_harvested") == "0" else "not harvested")
        rows.append({"journal code": j, "status": status, "records": got.get(j, 0), "note": note,
                     "harvested (UTC)": kv.get("harvested_utc", "")})
    df = pd.DataFrame(rows)
    out = ["## 1. Harvest", "",
           f"OAI-PMH, `{src['metadata_prefix']}`, all journals of the source `{src['name']}` "
           f"({len(journals)} journals; {int((df['status'] != 'excluded').sum())} included, "
           f"{int((df['status'] == 'excluded').sum())} excluded, "
           f"{int((df['status'] == 'included (partial)').sum())} partial). Raw responses are stored unchanged.", "",
           md_table(df), "",
           "Sources: `config.yaml` (dataset.sources), `results/tables/harvest_summary_uomosul_ojs.csv`, "
           "`data/raw/oai/*/HARVEST_INFO.txt`, harvest log.", ""]
    return out


def cleaning_section(root: Path, cfg: dict) -> list[str]:
    st = json.loads((root / "results/tables/dataset_stats.json").read_text(encoding="utf-8"))
    pl = st.get("pipeline", {})
    fc = st.get("field_completeness_pct", {})
    out = ["## 2. Cleaning and the final collection", "",
           "| step | records |", "|---|---|",
           f"| loaded from the harvest | {pl.get('loaded')} |",
           f"| dropped: empty title or placeholder value | {pl.get('dropped_empty_or_placeholder')} |",
           f"| dropped: duplicates (same DOI/URL, or same normalised title within a journal) | {pl.get('duplicates_removed')} |",
           f"| final collection | {pl.get('final')} |", "",
           f"Placeholder values treated as empty: {', '.join(cfg['dataset']['placeholder_values'])}; "
           f"minimum title length {cfg['dataset']['min_title_chars']} characters. Text was cleaned of HTML tags, "
           "normalised to Unicode NFKC, and Persian yeh/keheh (U+06CC, U+06A9) were mapped to Arabic yeh/kaf "
           "(U+064A, U+0643); platform boiler-plate keywords were removed (`librag/dataset.py`, `librag/text.py`).", "",
           "| field | records with the field (%) |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in fc.items()]
    out += ["", f"Bilingual (Arabic and English) titles: {st.get('bilingual_ar_en_pct')}%; Arabic title: "
                f"{st.get('has_arabic_title_pct')}%; English title: {st.get('has_english_title_pct')}%; years "
                f"{st.get('year_min')}–{st.get('year_max')}. Records per journal: `dataset_stats.json` (collections).",
            "", "Source: `results/tables/dataset_stats.json`.", ""]
    return out


def retrieval_section(root: Path, cfg: dict) -> list[str]:
    r = cfg["retrieval"]
    out = ["## 3. Retrieval systems", "",
           f"- Text templates: " + "; ".join(f"`{k}` = {'+'.join(v)}" for k, v in cfg["templates"].items())
           + f"; main template `{cfg['main_template']}`, metadata languages `{cfg['template_languages']}`.",
           f"- BM25: k1 = {r['bm25']['k1']}, b = {r['bm25']['b']}, stemming = {r['bm25']['stem']}, "
           f"stop-word removal = {r['bm25']['remove_stopwords']}. Arabic: diacritics and tatweel removed, letter "
           "variants unified, light10 stemming; English: Porter stemming (`librag/text.py`).",
           f"- Dense: exact cosine search over normalised embeddings; sequence cap {r['max_seq_length_cap']} tokens; "
           "models and prefixes in `config.yaml` (retrieval.dense_models).",
           f"- Hybrid: reciprocal rank fusion, k = {r['hybrid']['rrf_k']}, fusion depth {r['hybrid']['fuse_depth']}.",
           f"- Run depth {r['depth']}. Field ablation: {', '.join(cfg['ablation_templates'])} for "
           f"{', '.join(cfg['ablation_models'])}; language ablation: {', '.join(cfg['language_ablation'])}.", ""]
    ib = read_csv(root / "results/tables/index_build.csv")
    if ib is not None:
        sub = ib[(ib["template"] == cfg["main_template"]) & (ib["languages"] == cfg["template_languages"])]
        cols = ["system", "max_seq_length", "truncated_pct", "token_length_median", "token_length_p95",
                "encode_seconds", "build_seconds"]
        sub = sub[[c for c in cols if c in sub.columns]].round(2)
        out += ["Index build for the main template (truncation is the share of records longer than the model's "
                "sequence limit):", "", md_table(sub), "", "Source: `results/tables/index_build.csv`.", ""]
    return out


def generation_section(root: Path, cfg: dict) -> list[str]:
    g = cfg["generation"]
    out = ["## 4. Generation and citation verification", "",
           f"- Models (Ollama): {', '.join(g['models'])}.",
           f"- Options: {json.dumps(g['options'])}; retriever `{g['rag_retriever']}` (fixed before the experiment); "
           f"k = {g['k']}; abstracts truncated to {g['max_abstract_chars']} characters.",
           f"- Conditions: {', '.join(g['conditions'])}; k ablation {g['k_ablation']} for "
           f"{', '.join(g['k_ablation_models'])}. Prompt texts: `librag/generation/`.",
           f"- Detector: title similarity threshold {g['verification']['title_threshold']} "
           f"(sensitivity {g['verification']['sensitivity_thresholds']}), {g['verification']['candidate_pool']} "
           "catalogue candidates per title (`librag/generation/`).", ""]
    p = root / "results/generation/responses_verified.jsonl"
    if p.exists():
        n = changed = truncated = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                n += 1
                a, b = r["verification"], r["verification_at_generation"]
                if [i["label"] for i in a["items"]] != [i["label"] for i in b["items"]] or a["abstained"] != b["abstained"]:
                    changed += 1
                truncated += (r.get("llm") or {}).get("done_reason") == "length"
        out += [f"- Responses: {n}. Responses whose labels changed when the detector rules were refined (before "
                f"the rules were frozen and before any human evaluation): {changed}. Responses that reached the "
                f"output cap of {g['options']['num_predict']} tokens: {truncated} ({100 * truncated / n:.1f}%); "
                "their incomplete last line is not counted as a citation.", "",
                "Source: `results/generation/responses_verified.jsonl` (fields `verification`, "
                "`verification_at_generation`, `llm.done_reason`).", ""]
    ts = read_csv(root / "results/tables/generation_threshold_sensitivity.csv")
    if ts is not None:
        piv = ts.pivot_table(index=["model", "condition"], columns="threshold", values="hallucination_rate").round(3)
        piv.columns = [f"threshold {c}" for c in piv.columns]
        out += ["Ungrounded-citation rate (answered queries) at three similarity thresholds:", "",
                md_table(piv.reset_index()), "", "Source: `results/tables/generation_threshold_sensitivity.csv`.", ""]
    return out


def queries_section(root: Path, cfg: dict) -> list[str]:
    out = ["## 5. Test queries", ""]
    q = pd.read_excel(root / cfg["paths"]["queries_file"])
    out += ["Written by two librarians (codes in `written_by`) following `docs/annotation_guidelines.md`; "
            "no patron data.", "",
            md_table(pd.crosstab(q["type"], q["lang"]).reset_index()), "",
            md_table(pd.crosstab(q["phrasing"].replace({"\u062d\u0648\u0627\u0631\u064a": "conversational",
                                                       "\u0643\u0644\u0645\u0627\u062a \u0645\u0641\u062a\u0627\u062d\u064a\u0629": "keywords"}),
                                 q["lang"]).reset_index()), "",
            f"Queries per journal (journal field): {q['journal'].value_counts().to_dict()}.", ""]
    cc = read_csv(root / "results/tables/query_copy_check.csv")
    if cc is not None:
        fl = cc[cc["flagged"].astype(str) == "True"]
        acc = cfg["evaluation"].get("copy_check_accepted", [])
        out += [f"Copied-phrase screen (a run of shared content words with a title or keyword list): "
                f"{len(fl)} flagged ({', '.join(fl['qid'])}); kept after review: {', '.join(acc)}.", "",
                "Source: `data/queries/queries.xlsx`, `results/tables/query_copy_check.csv`.", ""]
    return out


def judgements_section(root: Path, cfg: dict) -> list[str]:
    e = cfg["evaluation"]
    out = ["## 6. Relevance judgements", "",
           f"- Pool: top {e['pool_depth']} of {', '.join(e['pool_systems'])}; excluded from the pool of Arabic "
           f"queries: {', '.join(e['pool_exclude'].get('ar', []))}; known-item queries are not pooled (their target "
           f"is the only relevant record). Scale {e['relevance_scale']}; decision rules R1–R7 in "
           "`docs/annotation_guidelines.md`.", ""]
    pm = read_csv(root / "data/annotation/pool_manifest.csv")
    if pm is not None:
        out += [f"- Pooled pairs: {len(pm)} for {pm['qid'].nunique()} queries "
                f"({len(pm) / pm['qid'].nunique():.1f} per query). Source: `data/annotation/pool_manifest.csv`.", ""]
    rows = []
    for name, p in [("pilot 1", "data/annotation/pilot1/agreement_pilot1.csv"),
                    ("pilot 2 (after rules R1–R7)", "data/annotation/pilot2/agreement_pilot2.csv"),
                    ("test set", "results/tables/agreement.csv")]:
        a = read_csv(root / p)
        if a is not None:
            r = a.iloc[0]
            rows.append({"round": name, "pairs": int(r["n_pairs"]), "kappa (binary)": round(r["kappa_binary"], 2),
                         "kappa (linear weighted)": round(r["kappa_weighted_linear"], 2),
                         "raw agreement %": round(100 * r["percent_agreement"], 1), "source": f"`{p}`"})
    if rows:
        out += ["Agreement between the two librarians:", "", md_table(pd.DataFrame(rows)), ""]
    return out


def human_eval_section(root: Path, cfg: dict) -> list[str]:
    out = ["## 7. Human validation of the citation detector", ""]
    des = read_csv(root / "results/generation/human_eval_design.csv")
    if des is not None:
        out += ["Stratified sample, fixed quota per automatic label, re-weighted by label size:", "",
                md_table(des), "",
                f"A second evaluator judged {cfg['evaluation']['human_eval']['second_annotator_items']} items. "
                "Per item: in the supplied records, in the catalogue, exists elsewhere, authors correct, description "
                "faithful. Sources: `results/generation/human_eval_*.xlsx`, `human_eval_design.csv`; results in "
                "`results/tables/detector_validation.csv`, `human_agreement.csv`, `fabrication_breakdown.csv`.", ""]
    return out


def extra_results_section(root: Path) -> list[str]:
    out = ["## 8. Results reported in the article only in summary", ""]
    for title, p in [("Retrieval by query type and language (exploratory)", "results/tables/retrieval_by_subgroup.csv"),
                     ("Number of retrieved records k", "results/tables/generation_k_ablation.csv"),
                     ("Abstention by presence of a relevant record in the context",
                      "results/tables/generation_abstention_by_context.csv")]:
        df = read_csv(root / p)
        if df is not None:
            out += [f"### {title}", "", md_table(df), "", f"Source: `{p}`.", ""]
    return out


def main() -> None:
    cfg, _, log = init(parser(__doc__))
    root = Path(cfg["_root"])
    parts = ["# Implementation details", "",
             "Generated by `scripts/11_make_details.py` from the configuration, logs and result files of this "
             "repository. The article summarises these details; this file gives them in full.", ""]
    for fn in (harvest_section, cleaning_section, retrieval_section, generation_section, queries_section,
               judgements_section, human_eval_section):
        try:
            parts += fn(root, cfg)
        except FileNotFoundError as exc:
            log.info(f"skipped {fn.__name__}: {exc}")
    parts += extra_results_section(root)
    out = root / "docs" / "DETAILS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()
