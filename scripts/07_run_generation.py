"""Step 7 — run every local LLM under every condition (RQ3) and log raw outputs.

    ollama pull llama3.2:1b   (and the other models listed in config.yaml)
    python scripts/07_run_generation.py
    python scripts/07_run_generation.py --models qwen2.5:7b --limit 10   # quick check

Design: models × {norag_plain, norag_grounded, rag_plain, rag_grounded} at k = generation.k,
plus a k-ablation (rag_grounded, k ∈ k_ablation) for k_ablation_models.
Decoding is deterministic (temperature 0, fixed seed). Every response, its
retrieved context, timings and automatic verification are appended to
results/generation/responses.jsonl; the run resumes where it stopped.
"""
from __future__ import annotations

import json

from _common import init, load_records, parser, path, results_path

from librag.config import read_json, write_json
from librag.eval.io import read_queries
from librag.generation.llm import OllamaClient
from librag.generation.verify import CatalogMatcher
from librag.pipeline import RAGPipeline
from librag.retrieval import SystemFactory


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--models", nargs="*", help="subset of generation.models")
    ap.add_argument("--limit", type=int, help="only the first N queries (smoke test)")
    cfg, args, log = init(ap)
    g = cfg["generation"]
    records = load_records(cfg)
    queries = read_queries(path(cfg, "queries_file"))
    if args.limit:
        queries = queries.head(args.limit)

    out_file = results_path(cfg, "generation", "responses.jsonl")
    done = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                done.add((d["model"], d["condition"], d["k"], d["qid"]))
    log.info(f"{len(done)} responses already logged (resuming)")

    factory = SystemFactory(cfg)
    matcher = CatalogMatcher(records, g["verification"]["candidate_pool"])
    models = args.models or g["models"]
    model_meta = {}

    jobs = [(m, c, g["k"]) for m in models for c in g["conditions"]]
    jobs += [(m, "rag_grounded", k) for m in models if m in g["k_ablation_models"]
             for k in g["k_ablation"] if k != g["k"]]

    def remaining(model: str) -> int:
        return sum(1 for m, c, k in jobs if m == model for q in queries["qid"]
                   if (model, c, (k if c.startswith("rag") else 0), q) not in done)

    todo_by_model = {m: remaining(m) for m in models}
    if not any(todo_by_model.values()):
        log.info(f"all {len(done)} responses are already logged — nothing to run. Next step: 08_eval_generation.py")
        return
    log.info("responses still to run: " + ", ".join(f"{m}: {n}" for m, n in todo_by_model.items()))

    # Retrieve once for every query, then free the encoder's GPU memory so that each
    # LLM runs entirely on the 8 GB card (otherwise Ollama offloads layers to the CPU
    # and the measured latencies would be distorted).
    for q in queries["query"]:
        factory.search(g["rag_retriever"], q, max(g["k_ablation"] + [g["k"]]))
    factory.release_gpu()
    log.info(f"retrieval cached for {len(queries)} queries; encoder unloaded from GPU")

    with open(out_file, "a", encoding="utf-8") as fh:
        for model in models:
            if not todo_by_model[model]:
                log.info(f"{model}: complete, skipped")
                continue
            client = OllamaClient(g["ollama_url"], model, g["options"], g["keep_alive"], g["timeout_seconds"])
            try:
                info = client.model_info()
            except Exception as exc:
                raise SystemExit(f"model {model} not available in Ollama ({exc}). Run:  ollama pull {model}")
            info["cold_load_seconds"] = client.warm_up()
            model_meta[model] = info
            log.info(f"{model}: {info}")
            pipe = RAGPipeline(records, client, factory, g["rag_retriever"], matcher,
                               g["verification"]["title_threshold"], g["max_abstract_chars"])
            for m, condition, k in jobs:
                if m != model:
                    continue
                todo = [q for q in queries.itertuples() if (model, condition, (k if condition.startswith("rag") else 0), q.qid) not in done]
                log.info(f"{model} | {condition} | k={k}: {len(todo)} queries to run")
                for q in todo:
                    res = pipe.answer(q.query, condition, k, q.lang)
                    row = {"model": model, "qid": q.qid, "query": q.query, "lang": q.lang, "type": q.type,
                           "answerable": bool(q.answerable), **res}
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
            client.unload()                       # free VRAM before the next model is loaded
    meta_file = results_path(cfg, "generation", "models.json")
    if meta_file.exists():
        model_meta = {**read_json(meta_file), **model_meta}
    write_json(model_meta, meta_file)
    log.info(f"responses → {out_file}")


if __name__ == "__main__":
    main()
