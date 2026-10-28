"""End-to-end smoke test of scripts 01–10 with a fake encoder and a fake LLM.

It exercises every script on a tiny SYNTHETIC test collection so that code paths,
file formats and tables are checked without a GPU, model downloads or Ollama.
Nothing produced here is research data.

    python -m unittest tests.test_pipeline_e2e -v
"""
from __future__ import annotations

import hashlib
import random
import re
import runpy
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


# ------------------------------------------------------------------ fakes
class _FakeTokenizer:
    def __call__(self, texts, add_special_tokens=True, truncation=False):
        return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}


class _FakeST:
    def __init__(self, model_id, device=None):
        self.max_seq_length = 256 if "MiniLM" in model_id else 8192
        self.tokenizer = _FakeTokenizer()

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False):
        from librag.text import analyze

        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in analyze(t):
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


def _fake_chat(self, system, user):
    titles = re.findall(r"^\[(\d+)\] Title: ([^|/]+?)(?: / [^|]+)? \| Authors: ([^|;]+)", user, flags=re.M)
    lines = [f"- [{n}] {t.strip()} — {a.strip()}: relevant to the request." for n, t, a in titles[:2]]
    if "ONLY" not in system:
        lines.append("- An Invented Handbook of Imaginary Studies — Nobody Real: fabricated.")
    text = "\n".join(lines) if lines else "NO_RELEVANT_RECORDS"
    return {"text": text, "wall_s": 0.1, "total_s": 0.1, "load_s": 0.0, "prompt_tokens": 100, "prompt_eval_s": 0.01,
            "output_tokens": 40, "eval_s": 0.09, "tokens_per_s": 444.0, "done_reason": "stop"}


TOPICS = {
    "water": ("water quality river pollution", "جودة المياه تلوث الانهار"),
    "medai": ("artificial intelligence medical diagnosis", "الذكاء الاصطناعي التشخيص الطبي"),
    "libs": ("library cataloguing metadata standards", "الفهرسة الوصفية في المكتبات"),
    "agri": ("wheat agriculture irrigation yield", "زراعة القمح الري الانتاجية"),
}


def _make_fixture(tmp: Path) -> Path:
    rng = random.Random(0)
    rows = []
    for topic, (en, ar) in TOPICS.items():
        for i in range(10):
            arabic = i % 2 == 0
            rows.append({"ID": f"{topic}-{i:02d}",
                         "Title": f"{ar} دراسة رقم {i}" if arabic else f"Study {i} on {en}",
                         "Author": f"Author{rng.randint(1, 99)}, Test",
                         "Subjects": ";".join((ar if arabic else en).split()[:2]),
                         "Abstract": (f"ملخص عن {ar}" if arabic else f"An abstract about {en}.") if i % 3 else "",
                         "Language": "ar" if arabic else "en", "Year": str(2010 + i)})
    pd.DataFrame(rows).to_csv(tmp / "fixture_catalog.csv", index=False, encoding="utf-8-sig")
    q = [("q1", "books about river water pollution", "en", "topical", 1),
         ("q2", "artificial intelligence for medical diagnosis", "en", "topical", 1),
         ("q3", "الفهرسة الوصفية في المكتبات", "ar", "topical", 1),
         ("q4", "زراعة القمح والري", "ar", "topical", 1),
         ("q5", "metadata standards for cataloguing", "en", "exploratory", 1),
         ("q6", "تلوث الانهار", "ar", "topical", 1),
         ("q7", "quantum chromodynamics lattice simulations", "en", "topical", 0),
         ("q8", "تاريخ الفن البيزنطي", "ar", "topical", 0)]
    q = [row + ("",) for row in q]
    q.append(("q9", "that third study on irrigating wheat", "en", "known_item", 1,
              "Study 3 on wheat agriculture irrigation yield"))
    (tmp / "data" / "queries").mkdir(parents=True)
    pd.DataFrame(q, columns=["qid", "query", "lang", "type", "answerable", "target"]).to_excel(
        tmp / "data/queries/queries.xlsx", index=False)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["dataset"]["sources"] = [{"type": "csv", "name": "fx", "path": str(tmp / "fixture_catalog.csv"),
                                  "columns": {"id": "ID", "title": "Title", "authors": "Author", "subjects": "Subjects",
                                              "abstract": "Abstract", "language": "Language", "year": "Year"}}]
    for k in list(cfg["paths"]):
        cfg["paths"][k] = str(tmp / cfg["paths"][k])
    cfg["generation"]["models"] = ["fake:1b", "fake:3b"]
    cfg["generation"]["k_ablation_models"] = ["fake:1b"]
    cfg["evaluation"].update({"n_permutations": 200, "n_bootstrap": 200, "double_judged_fraction": 0.5,
                              "human_eval": {"items_per_label": {"not_in_catalog": 20, "grounded": 20,
                                                                 "misattributed": 10, "catalog_not_in_context": 10},
                                             "second_annotator_items": 25}})
    p = tmp / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


def _run(script: str, cfg: Path, *extra: str) -> None:
    sys.argv = [script, "--config", str(cfg), *extra]
    runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")


def _topic(doc_id: str) -> str:
    return doc_id.split(":", 1)[1].split("-")[0]


QUERY_TOPIC = {"q1": "water", "q2": "medai", "q3": "libs", "q4": "agri", "q5": "libs", "q6": "water", "q7": None, "q8": None,
               "q9": "agri"}


class EndToEnd(unittest.TestCase):
    def test_all_scripts(self):
        sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=_FakeST)
        from librag.generation import llm

        llm.OllamaClient.chat = _fake_chat
        llm.OllamaClient.model_info = lambda self: {"model": self.model, "quantization_level": "fake"}
        llm.OllamaClient.warm_up = lambda self: 0.5
        llm.OllamaClient.unload = lambda self: None

        tmp = Path(tempfile.mkdtemp())
        try:
            cfg = _make_fixture(tmp)
            _run("01_collect_records.py", cfg)
            _run("02_prepare_dataset.py", cfg)
            _run("03_build_indexes.py", cfg)
            _run("04_make_annotation_pool.py", cfg)

            ann = tmp / "data" / "annotation"
            pool = pd.read_excel(ann / "judgements_A1.xlsx", sheet_name="judgements", dtype=str)
            self.assertNotIn("q9", set(pool["qid"]))                   # known-item queries are not judged ...
            ki = pd.read_csv(ann / "known_item_qrels.csv", dtype=str)
            self.assertEqual(ki.set_index("qid").loc["q9", "doc_id"], "fx:agri-03")   # ... their target is the qrel
            manifest = pd.read_csv(ann / "pool_manifest.csv", dtype=str)
            arabic = manifest["qid"].isin(["q3", "q4", "q6", "q8"])
            self.assertFalse(manifest.loc[arabic, "systems"].str.contains("minilm_en").any())
            self.assertTrue(manifest.loc[~arabic, "systems"].str.contains("minilm_en").any())
            for a, flip in (("A1", 0.0), ("A2", 0.1)):
                rng = random.Random(1)
                wb = pd.read_excel(ann / f"judgements_{a}.xlsx", sheet_name="judgements", dtype=str)
                truth = [2 if QUERY_TOPIC[q] == _topic(d) else 0 for q, d in zip(wb["qid"], wb["doc_id"])]
                wb["judgement"] = [(1 if t == 0 else 0) if rng.random() < flip else t for t in truth]
                with pd.ExcelWriter(ann / f"judgements_{a}.xlsx") as xw:
                    wb.to_excel(xw, sheet_name="judgements", index=False)
            n1 = len(pd.read_excel(ann / "judgements_A1.xlsx", sheet_name="judgements"))
            n2 = len(pd.read_excel(ann / "judgements_A2.xlsx", sheet_name="judgements"))
            self.assertLess(n2, n1)                                     # A2 judges a subset of queries
            _run("05_import_judgements.py", cfg, "--partial")
            _run("05_import_judgements.py", cfg, "--apply-labels")
            self.assertTrue((tmp / "data" / "qrels" / "variants" / "lenient.txt").exists())
            qx = pd.read_excel(tmp / "data/queries/queries.xlsx", dtype=str)
            self.assertIn("answerable_written", qx.columns)
            qrels_q = {line.split()[0] for line in (tmp / "data" / "qrels" / "qrels.txt").read_text().splitlines()}
            self.assertEqual(len(qrels_q), 9)                           # single-judged and known-item queries are kept
            self.assertIn("q9 0 fx:agri-03 2", (tmp / "data" / "qrels" / "qrels.txt").read_text())
            _run("06_eval_retrieval.py", cfg)
            _run("07_run_generation.py", cfg)
            _run("08_eval_generation.py", cfg)

            res = tmp / "results"
            for a in ("A1", "A2"):
                wb = pd.read_excel(res / "generation" / f"human_eval_{a}.xlsx", dtype=str)
                fabricated = wb["item_line"].str.contains("Invented", na=False)
                wb["in_provided_records(y/n)"] = np.where(fabricated, "n", "y")
                wb["in_catalogue(y/n)"] = np.where(fabricated, "n", "y")
                wb["exists_elsewhere(y/n/unknown)"] = np.where(fabricated, "n", "")
                wb["author_correct(y/n/na)"] = "y"
                wb["description_faithful(y/partial/n)"] = "y"
                wb.to_excel(res / "generation" / f"human_eval_{a}.xlsx", index=False)
            _run("09_import_human_eval.py", cfg)
            _run("10_make_figures.py", cfg)

            t = res / "tables"
            sens = pd.read_csv(t / "retrieval_qrels_sensitivity.csv")
            self.assertIn("lenient", set(sens["qrels"]))
            main = pd.read_csv(t / "retrieval_main.csv")
            self.assertEqual(len(main), 7)                              # bm25 + 4 dense + 2 hybrid
            self.assertGreater(main.loc[main["system"] == "bm25", "ndcg@10"].item(), 0.5)
            gen = pd.read_csv(t / "generation_main.csv")
            plain = gen[gen["condition"] == "norag_plain"]["hallucination_rate"]
            grounded = gen[gen["condition"] == "rag_grounded"]["hallucination_rate"]
            self.assertTrue((plain == 1.0).all() and (grounded == 0.0).all())
            guard_t = pd.read_csv(t / "generation_guard.csv")
            self.assertTrue((guard_t["ungrounded_after"] == 0.0).all())
            val = pd.read_csv(t / "detector_validation.csv")
            self.assertEqual(val["recall"].item(), 1.0)
            for name in ("dataset_stats.json", "agreement.csv", "retrieval_significance.csv", "retrieval_field_ablation.csv",
                         "retrieval_crosslingual.csv", "retrieval_field_completeness.csv", "retrieval_latency.csv",
                         "generation_mcnemar.csv", "generation_k_ablation.csv", "generation_threshold_sensitivity.csv",
                         "fabrication_breakdown.csv", "faithfulness.csv"):
                self.assertTrue((t / name).exists(), name)
            self.assertTrue((res / "figures" / "fig_hallucination_abstention.png").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
