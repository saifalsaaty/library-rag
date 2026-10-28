"""Step 4 — build the depth-k judgement pool and one Excel workbook per annotator.

    python scripts/04_make_annotation_pool.py

Each query's pool is the union of the top `pool_depth` records returned by every
system in `evaluation.pool_systems`. Records are shuffled and system names are
hidden so annotators cannot tell which system retrieved what (blind judging).

Known-item queries: the queries file has a `target` column (the exact title of the
intended work), which is located in the collection. A target that cannot be found is
reported, because such a query has no correct answer. With
`evaluation.judge_known_items: false` (default) known-item queries are not sent for
judging: their target is the only relevant record (rules R5-R6) and is written to
known_item_qrels.csv. With `true`, their pools are judged and the target is added to
the pool when no system retrieved it.

`evaluation.pool_exclude` lists systems that do not contribute to the pools of queries
in a given language (e.g. the English-only MiniLM for Arabic queries).

Before pooling, queries that repeat six or more consecutive words of a record title are
reported and the script stops, so that no annotation time is spent on them. Rewrite those
queries, or pass --ignore-query-checks if a match is a harmless common phrase.
"""
from __future__ import annotations

import random

import pandas as pd

from _common import init, load_records, parser, path, results_path

from librag.dataset import copied_phrases, find_by_title
from librag.eval.agreement import double_judged_queries
from librag.eval.io import read_queries
from librag.eval.query_checks import copied_titles
from librag.retrieval import SystemFactory

INSTRUCTIONS = [
    ["Relevance judgement / تقييم الصلة"],
    [""],
    ["For each (query, record) pair enter a grade in column 'judgement':"],
    ["2 = highly relevant: a librarian would recommend this record for the request"],
    ["1 = partially relevant: related, useful as background or for part of the request"],
    ["0 = not relevant"],
    [""],
    ["لكل زوج (استعلام، سجل) أدخل درجة في عمود judgement:"],
    ["2 = وثيق الصلة: يوصي به أمين المكتبة لهذا الطلب"],
    ["1 = صلة جزئية: ذو علاقة ومفيد كخلفية أو لجزء من الطلب"],
    ["0 = غير ذي صلة"],
    [""],
    ["قواعد القرار / Decision rules"],
    ["R1  الموضوع الرئيس للسجل هو المطلوب نفسه ← 2"],
    ["R2  الموضوع نفسه لكن بفئة أو سياق أو منهج مختلف، أو يغطي جزءاً من الطلب ← 1"],
    ["R3  يشترك مع الطلب في كلمات لا في المعنى ← 0"],
    ["R4  المجال العام نفسه فقط ← 0"],
    ["R5  استعلام عمل معروف (known_item): العمل المذكور في عمود target ← 2"],
    ["R6  استعلام عمل معروف: أي عمل آخر حتى لو في الموضوع نفسه ← 0"],
    ["R7  لا ملخص والعنوان غامض ← 0 واكتب vague في note"],
    ["السؤال الحاسم: هل أعطي هذا السجل للمستفيد الذي سأل؟  /  Would I hand this record to the user who asked?"],
    [""],
    ["قيّم كل صف على حدة ولا تملأ العمود بالسحب أو النسخ. خذ استراحة قصيرة كل 50 صفاً. سجّل الوقت الكلي."],
    ["Judge every row on its own; never drag or copy a grade down. Log the total time spent."],
    ["قيّم كل زوج باستقلالية ودون نقاش مع المقيّمين الآخرين. الإرشادات الكاملة في docs/annotation_guidelines.md"],
    ["Judge each pair independently; do not discuss with other annotators. Full guidelines: docs/annotation_guidelines.md"],
]


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--ignore-query-checks", action="store_true",
                    help="build the pool even if some queries repeat a record title")
    cfg, args, log = init(ap)
    e = cfg["evaluation"]
    queries = read_queries(path(cfg, "queries_file"))
    if queries.empty:
        raise SystemExit("the queries file has no queries yet (column 'query' is empty)")
    records = load_records(cfg)

    if "written_by" in queries.columns and queries["written_by"].isna().any():
        log.info(f"WARNING: written_by is empty for {queries.loc[queries['written_by'].isna(), 'qid'].tolist()}")
    copied = copied_titles(queries, records)
    for c in copied.itertuples():
        log.info(f"WARNING: {c.qid} repeats {c.words_copied} consecutive words of a record title "
                 f"({c.n_titles} title(s)), e.g. «{c.title[:90]}»")
    if len(copied) and not args.ignore_query_checks:
        raise SystemExit(
            f"{len(copied)} queries copy words from record titles: {copied['qid'].tolist()}.\n"
            "Rewrite them as a user would ask (docs/annotation_guidelines.md, section 1, rules 2 and 4) and run again.\n"
            "استعلامات تكرر كلمات من عناوين البحوث؛ أعد صياغتها كما يسأل المستفيد ثم أعد التشغيل.\n"
            "If a match is only a common phrase, run again with --ignore-query-checks")
    factory = SystemFactory(cfg)
    qmap = dict(zip(queries["qid"], queries["query"]))

    accepted = {str(q) for q in (e.get("copy_check_accepted") or [])}
    checks = []
    for qid, text in qmap.items():
        hits = copied_phrases(records, text)
        for n, doc, field, phrase in hits[:1]:
            if qid in accepted:
                log.info(f"{qid} shares '{phrase}' with the {field} of record {doc} — accepted after review")
            else:
                log.info(f"WARNING: {qid} shares '{phrase}' with the {field} of record {doc} "
                         f"— rephrase it in the user's own words (docs/annotation_guidelines.md, section 1)")
        checks.append({"qid": qid, "query": text, "flagged": bool(hits),
                       "accepted_after_review": bool(hits) and qid in accepted,
                       "shared_content_words": hits[0][0] if hits else 0,
                       "field": hits[0][2] if hits else "", "record": hits[0][1] if hits else "",
                       "shared_phrase": hits[0][3] if hits else ""})
    pd.DataFrame(checks).to_csv(results_path(cfg, "tables", "query_copy_check.csv"), index=False, encoding="utf-8-sig")
    n_flag = sum(c["flagged"] for c in checks)
    n_acc = sum(c["accepted_after_review"] for c in checks)
    log.info(f"copied-phrase screen: {n_flag}/{len(checks)} queries flagged, {n_acc} accepted after review "
             f"(journal name, standard term or conversational formula), {n_flag - n_acc} to rephrase")

    out_dir = path(cfg, "annotation_dir")
    judge_ki = bool(e.get("judge_known_items", False))
    auto_ki = [] if judge_ki else list(queries.loc[queries["type"] == "known_item", "qid"])
    judged = queries[~queries["qid"].isin(auto_ki)]
    pool_qmap = {q: qmap[q] for q in judged["qid"]}
    qlang = dict(zip(queries["qid"], queries["lang"]))
    exclude = {lang: set(v or []) for lang, v in (e.get("pool_exclude") or {}).items()}

    contributions: dict[tuple[str, str], list[str]] = {}
    for system in e["pool_systems"]:
        run = factory.run(system, pool_qmap, depth=e["pool_depth"])
        for qid, ranked in run.items():
            if system in exclude.get(qlang[qid], ()):
                continue
            for doc, _ in ranked[: e["pool_depth"]]:
                contributions.setdefault((qid, doc), []).append(system)
        log.info(f"pooled {system}")
    for lang, systems in exclude.items():
        if systems:
            log.info(f"not pooled for {lang} queries: {sorted(systems)}")

    targets: dict[str, str] = {}
    target_titles: dict[str, str] = {}
    if "target" in queries.columns:
        for qid, row in queries.iterrows():
            title = str(row["target"]).strip() if isinstance(row["target"], str) else ""
            if not title:
                if row["type"] == "known_item":
                    log.info(f"WARNING: known-item query {qid} has no target title")
                continue
            target_titles[qid] = title
            doc, sim = find_by_title(records, title)
            if doc is None:
                log.info(f"WARNING: target of {qid} not found in the collection (best title similarity {sim:.2f}). "
                         f"Choose a work from the harvested journals or relabel the query.")
                continue
            targets[qid] = doc
            if qid in pool_qmap and (qid, doc) not in contributions:
                contributions[(qid, doc)] = ["target"]
                log.info(f"{qid}: target not retrieved by any system — added to the pool")
        log.info(f"known-item targets located: {len(targets)}/{len(target_titles)}")
    if auto_ki:
        ki = pd.DataFrame([{"qid": q, "doc_id": targets[q], "grade": 2, "target": target_titles.get(q, "")}
                           for q in auto_ki if q in targets], columns=["qid", "doc_id", "grade", "target"])
        ki.to_csv(out_dir / "known_item_qrels.csv", index=False, encoding="utf-8-sig")
        log.info(f"{len(auto_ki)} known-item queries are not sent for judging: their target is the only "
                 f"relevant record (R5-R6) → {out_dir / 'known_item_qrels.csv'} ({len(ki)} targets)")

    rng = random.Random(cfg["project"]["seed"])
    rows = []
    for qid in judged["qid"]:
        docs = [d for (q, d) in contributions if q == qid]
        rng.shuffle(docs)
        for d in docs:
            r = records[d]
            rows.append({
                "qid": qid, "query": qmap[qid], "target": target_titles.get(qid, ""), "doc_id": d,
                "title": " / ".join(r.titles()), "authors": "; ".join(r.authors[:6]),
                "year": r.year, "subjects": "; ".join(r.subjects[:12]),
                "abstract": " ".join(r.abstracts())[:3000],
                "judgement": "", "note": "",
            })
    pool = pd.DataFrame(rows)
    manifest = pd.DataFrame([{"qid": q, "doc_id": d, "systems": "|".join(s)} for (q, d), s in contributions.items()])
    manifest.to_csv(out_dir / "pool_manifest.csv", index=False, encoding="utf-8-sig")

    fraction = float(e.get("double_judged_fraction", 1.0))
    shared = double_judged_queries(judged, fraction, cfg["project"]["seed"])
    if fraction < 1:
        pd.DataFrame({"qid": shared}).to_csv(out_dir / "double_judged_queries.csv", index=False, encoding="utf-8-sig")
        log.info(f"double judging: {len(shared)}/{len(judged)} judged queries go to every annotator; "
                 f"the rest to {e['annotators'][0]} only (stratified by language, type and answerability)")

    for i, ann in enumerate(e["annotators"]):
        sheet = pool if i == 0 else pool[pool["qid"].isin(shared)]
        target = out_dir / f"judgements_{ann}.xlsx"
        if target.exists():
            log.info(f"{target} exists — not overwritten (delete it to regenerate)")
            continue
        with pd.ExcelWriter(target, engine="openpyxl") as xw:
            pd.DataFrame(INSTRUCTIONS).to_excel(xw, sheet_name="instructions", index=False, header=False)
            sheet.to_excel(xw, sheet_name="judgements", index=False)
            ws = xw.sheets["judgements"]
            from openpyxl.styles import Alignment
            from openpyxl.utils import get_column_letter
            from openpyxl.worksheet.datavalidation import DataValidation

            widths = {"qid": 8, "query": 40, "target": 30, "doc_id": 18, "title": 50, "authors": 25, "year": 6,
                      "subjects": 35, "abstract": 80, "judgement": 10, "note": 20}
            letter = {c: get_column_letter(i + 1) for i, c in enumerate(pool.columns)}
            for col, w in widths.items():
                ws.column_dimensions[letter[col]].width = w
            if not target_titles:
                ws.column_dimensions[letter["target"]].hidden = True

            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
            dv = DataValidation(type="list", formula1='"0,1,2"', allow_blank=True)
            ws.add_data_validation(dv)
            dv.add(f"{letter['judgement']}2:{letter['judgement']}{len(sheet) + 1}")
            ws.freeze_panes = "D2"
        log.info(f"annotator workbook → {target}  ({len(sheet)} pairs)")
    log.info(f"{len(judged)} of {len(queries)} queries sent for judging, {len(pool)} pooled pairs "
             f"(mean {len(pool) / max(len(judged), 1):.1f} records per query)")


if __name__ == "__main__":
    main()
