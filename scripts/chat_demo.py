"""Interactive demo of the evaluated system (replaces rag_core.py / app.py).

    python scripts/chat_demo.py
    python scripts/chat_demo.py --model qwen2.5:7b --k 3 --condition rag_grounded

Shows the retrieved records, the raw model answer, the verification labels and
the verified answer, so librarians can audit every recommendation.
"""
from __future__ import annotations

from _common import init, load_records, parser

from librag.generation.llm import OllamaClient
from librag.generation.verify import CatalogMatcher
from librag.pipeline import RAGPipeline
from librag.retrieval import SystemFactory


def main() -> None:
    ap = parser(__doc__)
    ap.add_argument("--model")
    ap.add_argument("--k", type=int)
    ap.add_argument("--condition", default="rag_grounded")
    cfg, args, _ = init(ap)
    g = cfg["generation"]
    records = load_records(cfg)
    model = args.model or g["models"][-1]
    llm = OllamaClient(g["ollama_url"], model, g["options"], g["keep_alive"], g["timeout_seconds"])
    pipe = RAGPipeline(records, llm, SystemFactory(cfg), g["rag_retriever"],
                       CatalogMatcher(records, g["verification"]["candidate_pool"]),
                       g["verification"]["title_threshold"], g["max_abstract_chars"])
    print(f"Model: {model} | retriever: {g['rag_retriever']} | records: {len(records)}  (empty line to quit)")
    while True:
        q = input("\nQuery / الاستعلام: ").strip()
        if not q:
            break
        res = pipe.answer(q, args.condition, args.k or g["k"])
        print("\n--- Retrieved records ---")
        for n, d in enumerate(res["context_ids"], start=1):
            r = records[d]
            print(f"[{n}] {r.primary_title()} — {'; '.join(r.authors[:3])} ({r.year or 'n.d.'})")
        print("\n--- Model answer ---\n" + res["response"])
        print("\n--- Verification ---")
        for it in res["verification"]["items"]:
            print(f"  {it['label']:24s} {it['title'][:70]}")
        if res["guarded_response"] is not None:
            print("\n--- Verified answer ---\n" + res["guarded_response"])
        print(f"\n(retrieval {res['retrieval_s']:.3f}s, generation {res['llm']['total_s']:.2f}s)")


if __name__ == "__main__":
    main()
