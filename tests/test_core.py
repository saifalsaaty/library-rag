"""Unit tests for text processing, loaders, BM25, metrics, statistics and verification.

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIX = ROOT / "tests" / "fixtures"

from librag.dataset import copied_phrases, deduplicate, describe, find_by_title, record_text  # noqa: E402
from librag.eval import metrics as M  # noqa: E402
from librag.eval.agreement import combine_judgements, double_judged_queries, fleiss_kappa  # noqa: E402
from librag.eval.query_checks import copied_titles, longest_common_run  # noqa: E402
from librag.eval.stats import holm, mcnemar_exact, randomization_test, wilson_ci  # noqa: E402
from librag.generation.verify import CatalogMatcher, guard, parse_items, verify  # noqa: E402
from librag.loaders.marc import read_marcxml  # noqa: E402
from librag.loaders.oai import parse_page  # noqa: E402
from librag.retrieval.bm25 import BM25  # noqa: E402
from librag.retrieval.hybrid import rrf  # noqa: E402
from librag.schema import Record  # noqa: E402
from librag.text import analyze, ar_light_stem, detect_lang, normalize, title_similarity  # noqa: E402


class TextTests(unittest.TestCase):
    def test_arabic_normalisation(self):
        self.assertEqual(normalize("إِسْلامُ"), "اسلام")
        self.assertEqual(normalize("مكتبة"), "مكتبه")
        self.assertEqual(normalize("مستشفى"), "مستشفي")

    def test_light10(self):
        self.assertEqual(ar_light_stem(normalize("والمكتبات")), "مكتب")
        self.assertEqual(ar_light_stem(normalize("الطلاب")), "طلاب")

    def test_analyze_removes_conversational_fillers(self):
        self.assertEqual(analyze("I am looking for books about libraries", stem=False), ["libraries"])
        self.assertNotIn("ابحث", analyze("أبحث عن كتب في الفهرسة", stem=False))

    def test_detect_lang(self):
        self.assertEqual(detect_lang("جودة المياه"), "ar")
        self.assertEqual(detect_lang("water quality"), "en")
        self.assertIsNone(detect_lang("12"))

    def test_title_similarity(self):
        self.assertGreaterEqual(title_similarity("AI in Healthcare", "AI in Healthcare: Fundamentals and Applications"), 0.85)
        self.assertLess(title_similarity("Deep learning", "Water quality of the Tigris river"), 0.5)
        self.assertEqual(title_similarity("الذكاء الاصطناعي في الطب", "الذّكاء الإصطناعي في الطب"), 1.0)


class KnownItemTargetTests(unittest.TestCase):
    def test_find_by_title(self):
        recs = {
            "a": Record(id="a", source="t", title={"ar": "القانون الدولي الانساني وتطبيقاته على النزاعات المسلحة في العراق",
                                                   "en": "International humanitarian law and its applications"}),
            "b": Record(id="b", source="t", title={"ar": "المرتزقة في القانون الدولي الانساني"}),
        }
        self.assertEqual(find_by_title(recs, "القانون الدولي الإنساني وتطبيقاته على النزاعات المسلحة في العراق")[0], "a")
        self.assertEqual(find_by_title(recs, "International Humanitarian Law and its Applications")[0], "a")
        self.assertIsNone(find_by_title(recs, "Evaluation of cytogenetic damage induced by Amlodipine in vitro")[0])

    def test_copied_phrases(self):
        recs = {
            "a": Record(id="a", source="t", title={"en": "An Efficient Algorithm for Eye Detection in Faces Images"}),
            "b": Record(id="b", source="t", title={"en": "A survey"},
                        subjects=["HCSE, Emotion Recognition, User Interfaces, Software Engineering"]),
        }
        hits = copied_phrases(recs, "I am writing a thesis on Eye Detection in Faces Images, what can I read")
        self.assertEqual(hits[0][1:3], ("a", "title"))
        self.assertEqual(copied_phrases(recs, "Emotion Recognition, User Interfaces, Software Engineering")[0][2], "subjects")
        self.assertEqual(copied_phrases(recs, "finding papers about detecting eyes in photographs"), [])

    def test_copied_title_check(self):
        import pandas as pd
        recs = {"a": Record(id="a", source="t", title={"en": "Efficiency of Two Rotary Retreatment Systems with and without XP-Endo Finisher"}),
                "b": Record(id="b", source="t", title={"ar": "فاعلية برنامج تعليمي قائم على الأنشطة الفنية في اكتساب المفاهيم"})}
        q = pd.DataFrame({"qid": ["x1", "x2", "x3"],
                          "query": ["Efficiency of two rotary retreatment systems published 2022",
                                    "a 2022 dental study comparing rotary files for removing old root fillings",
                                    "فاعلية برنامج تعليمي قائم على الأنشطة الفنية نشر قبل سنوات"]}).set_index("qid", drop=False)
        c = copied_titles(q, recs)
        self.assertEqual(sorted(c["qid"]), ["x1", "x3"])
        self.assertEqual(longest_common_run("a b c d".split(), "x b c d y".split()), 3)


class DoubleJudgingTests(unittest.TestCase):
    def test_stratified_query_sample(self):
        import pandas as pd

        q = pd.DataFrame({"qid": [f"Q{i:03d}" for i in range(1, 121)],
                          "lang": ["ar"] * 60 + ["en"] * 60,
                          "type": (["topical"] * 24 + ["exploratory"] * 18 + ["known_item"] * 18) * 2,
                          "answerable": ([True] * 48 + [False] * 12) * 2})
        shared = double_judged_queries(q, 0.3, seed=42)
        self.assertEqual(shared, double_judged_queries(q, 0.3, seed=42))           # reproducible
        self.assertTrue(30 <= len(shared) <= 42)
        sub = q[q["qid"].isin(shared)]
        self.assertEqual(set(sub["lang"]), {"ar", "en"})
        self.assertEqual(set(sub["type"]), {"topical", "exploratory", "known_item"})
        self.assertEqual(set(sub["answerable"]), {True, False})
        self.assertEqual(double_judged_queries(q, 1.0, seed=42), list(q["qid"]))

    def test_combine(self):
        import pandas as pd

        idx = pd.MultiIndex.from_tuples([("q1", "a"), ("q1", "b"), ("q2", "c")])
        a1 = pd.Series([2, 0, 1], index=idx)
        a2 = pd.Series([1, 0], index=idx[:2])                                       # q2 judged by A1 only
        overlap, final = combine_judgements({"A1": a1, "A2": a2}, "A1")
        self.assertEqual(len(overlap), 2)
        self.assertEqual(final.to_dict(), {("q1", "a"): 1, ("q1", "b"): 0, ("q2", "c"): 1})


class LoaderTests(unittest.TestCase):
    def test_oai_pages(self):
        recs, token, n = parse_page((FIX / "oai_page1.xml").read_bytes(), "fx")
        self.assertEqual((len(recs), token, n), (1, "TOKEN-2", 2))       # deleted record skipped
        r = recs[0]
        self.assertEqual(set(r.title), {"en", "ar"})
        self.assertEqual(r.subjects, ["water quality", "river", "جودة المياه", "نهر"])
        self.assertNotIn("<p>", r.abstract["en"])
        self.assertEqual((r.language, r.year, r.collection, r.rtype), ("en", 2024, "jrnl", "article"))
        recs2, token2, _ = parse_page((FIX / "oai_page2.xml").read_bytes(), "fx")
        self.assertIsNone(token2)
        self.assertEqual(recs2[0].language, "ar")

    def test_marcxml(self):
        recs = list(read_marcxml(FIX / "marc_sample.xml", "lib"))
        self.assertEqual(len(recs), 2)
        a = recs[0]
        self.assertEqual(a.title, {"ar": "كتاب اختبار في الفهرسة", "en": "A test book on cataloguing"})
        self.assertEqual((a.language, a.year, a.subjects, a.classification), ("ar", 2020, ["الفهرسة -- قواعد"], ["025.3"]))
        self.assertEqual(recs[1].authors, ["Tester, Ann"])

    def test_templates_and_dedup(self):
        r1 = Record("x:1", "x", {"en": "A title here"}, ["Smith, J"], ["s1"], {"en": "abstract text"}, "en", 2020)
        r2 = Record("x:2", "x", {"en": "A Title Here"}, [], [], {}, "en", 2020)
        self.assertEqual(record_text(r1, ["title", "subjects"]), "Title: A title here | Subjects: s1")
        unique, removed = deduplicate([r2, r1])
        self.assertEqual((len(unique), removed, unique[0].id), (1, 1, "x:1"))
        self.assertEqual(describe([r1, r2])["field_completeness_pct"]["abstract"], 50.0)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["d1", "d2", "d3"]
        self.texts = ["water quality of rivers", "deep learning for images", "river water pollution and water quality"]
        self.bm = BM25(k1=1.2, b=0.75, stem=False, remove_stopwords=False).fit(self.ids, self.texts)

    def test_bm25_matches_hand_computation(self):
        n, df, avgdl = 3, 2, (4 + 4 + 6) / 3
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        tf, dl = 1, 4
        expected = idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * dl / avgdl))
        self.assertAlmostEqual(self.bm.scores("quality")[0], expected, places=6)

    def test_bm25_ranking_and_zero_results(self):
        self.assertEqual([d for d, _ in self.bm.search("water quality")], ["d3", "d1"])
        self.assertEqual(self.bm.search("astronomy"), [])

    def test_rrf(self):
        fused = rrf([[("a", 3), ("b", 2)], [("b", 9), ("c", 1)]], k=60)
        self.assertEqual(fused[0][0], "b")
        self.assertAlmostEqual(fused[0][1], 1 / 62 + 1 / 61)


class MetricTests(unittest.TestCase):
    def test_metrics(self):
        rels = {"a": 2, "b": 0, "c": 1}
        ranked = ["x", "a", "b", "c"]
        self.assertAlmostEqual(M.precision_at(ranked, rels, 2), 0.5)
        self.assertAlmostEqual(M.recall_at(ranked, rels, 4), 1.0)
        self.assertAlmostEqual(M.rr_at(ranked, rels, 10), 0.5)
        dcg = 2 / math.log2(3) + 1 / math.log2(5)
        idcg = 2 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(M.ndcg_at(ranked, rels, 10), dcg / idcg)
        self.assertAlmostEqual(M.average_precision(ranked, rels), (1 / 2 + 2 / 4) / 2)
        self.assertAlmostEqual(M.judged_at(ranked, rels, 4), 0.75)

    def test_per_query_skips_unanswerable(self):
        pq = M.per_query({"q1": [("a", 1.0)], "q2": [("z", 1.0)]}, {"q1": {"a": 1}, "q2": {"z": 0}})
        self.assertEqual(list(pq.index), ["q1"])


class CondensedTests(unittest.TestCase):
    def test_condense_drops_unjudged_except_complete_queries(self):
        run = {"q1": [("x", 3.0), ("a", 2.0), ("b", 1.0)], "q2": [("y", 2.0), ("t", 1.0)]}
        qrels = {"q1": {"a": 2, "b": 0}, "q2": {"t": 2}}
        c = M.condense(run, qrels, complete={"q2"})
        self.assertEqual([d for d, _ in c["q1"]], ["a", "b"])           # unjudged "x" removed
        self.assertEqual([d for d, _ in c["q2"]], ["y", "t"])           # known-item query untouched
        self.assertGreater(M.ndcg_at(["a", "b"], qrels["q1"], 10), M.ndcg_at(["x", "a", "b"], qrels["q1"], 10))


class TruncationTests(unittest.TestCase):
    def test_last_item_of_truncated_response_is_dropped(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("g08", ROOT / "scripts" / "08_eval_generation.py")
        sys.path.insert(0, str(ROOT / "scripts"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        v = {"items": [{"label": "grounded"}, {"label": "not_in_catalog"}]}
        self.assertEqual(len(mod.scored_items(v, {"done_reason": "length"})), 1)
        self.assertEqual(len(mod.scored_items(v, {"done_reason": "stop"})), 2)


class StatsTests(unittest.TestCase):
    def test_randomization(self):
        self.assertEqual(randomization_test([0.5, 0.6], [0.5, 0.6]), 1.0)
        self.assertLess(randomization_test([1.0] * 30, [0.0] * 30, 2000), 0.01)

    def test_wilson(self):
        lo, hi = wilson_ci(0, 50)
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.0713, places=3)       # a 0/50 result is NOT "0%" with certainty

    def test_holm_and_mcnemar(self):
        adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["b"], 0.06)
        r = mcnemar_exact([1, 1, 1, 0], [0, 0, 0, 0])
        self.assertEqual((r["only_first"], r["only_second"]), (3, 0))
        self.assertAlmostEqual(r["p_value"], 0.25)

    def test_fleiss(self):
        self.assertAlmostEqual(fleiss_kappa([[3, 0], [0, 3], [3, 0]]), 1.0)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.recs = {
            "a": Record("a", "t", {"en": "Machine Learning for Medical Diagnosis"}, ["Hassan, Ali"], year=2021),
            "b": Record("b", "t", {"en": "Water Quality of the Tigris River", "ar": "جودة مياه نهر دجلة"}, ["Ahmed, Sara"]),
            "c": Record("c", "t", {"en": "Ethics of Algorithms in Public Institutions"}, ["Brown, Mike"]),
        }
        self.m = CatalogMatcher(self.recs)

    def test_labels(self):
        txt = ("Here are some resources:\n"
               "- [1] Machine Learning for Medical Diagnosis — Ali Hassan: diagnostic models.\n"
               "- [2] Ethics of Algorithms in Public Institutions — John Smith: bias.\n"
               "- Deep Learning in Radiology — Jane Doe: a classic.\n"
               "- \"Water Quality of the Tigris River\" by Sara Ahmed: hydrology.")
        v = verify(txt, ["a", "c"], self.m)
        self.assertEqual([i.label for i in v.items],
                         ["grounded", "misattributed", "not_in_catalog", "catalog_not_in_context"])
        out = guard(v, ["a", "c"], self.recs, "en")
        self.assertIn("Machine Learning for Medical Diagnosis", out)
        self.assertNotIn("Radiology", out)

    def test_arabic_and_cross_script_authors(self):
        v = verify("- [1] جودة مياه نهر دجلة — سارة أحمد: دراسة\n- [٢] كتاب غير موجود عن الزراعة — مؤلف: شرح", ["b", "a"], self.m)
        self.assertEqual([i.label for i in v.items], ["grounded", "not_in_catalog"])

    def test_abstention(self):
        self.assertTrue(verify("NO_RELEVANT_RECORDS", ["a"], self.m).abstained)
        self.assertTrue(verify("لا توجد سجلات ذات صلة.", [], self.m).abstained)
        self.assertFalse(verify("- [1] Machine Learning for Medical Diagnosis — Ali Hassan: x", ["a"], self.m).abstained)
        self.assertIn("relevant records", guard(verify("- Unknown Book Title Here — X: y", ["a"], self.m), ["a"], self.recs, "en"))

    def test_formats_seen_in_the_experiment(self):
        """Answer formats produced by the real models (October 2026) must not create false 'misattributed' labels."""
        recs = {
            "d": Record("d", "t", {"ar": "مرجعيات الذاكرة في قصص نجمان ياسين", "en": "Memory references in Najman Yassin stories"},
                        ["محمود, يمامة", "مال الله, وسن", "Mahmood, Yamama", "Mall Allah, Wasan"]),
            "e": Record("e", "t", {"en": "Effect of training program by using compound exercises on football players"},
                        ["Koshnaw, Jammil"]),
        }
        m = CatalogMatcher(recs)
        lines = [
            "[1] Title: مرجعيات الذاكرة في قصص نجمان ياسين / Memory references in Najman Yassin stories — "
            "Authors: Mahmood, Yamama; Mall Allah, Wasan: This record is relevant to the request.",
            "[1] مرجعيات الذاكرة في قصص نجمان ياسين / Memory references in Najman Yassin stories / محمود, يمامة — "
            "This record is relevant because it discusses memory.",
            "- Memory references in Najman Yassin stories | Authors: Mahmood, Yamama | Year: 2020 | Subjects: memory | A study.",
            "[2] Effect of training program by using compound exercises on football players — Authors: Koshnaw, Jammil: relevant",
        ]
        for line in lines:
            self.assertEqual([i.label for i in verify(line, ["d", "e"], m).items], ["grounded"], line)
        self.assertEqual(verify("[1] is the most relevant record for this request.\n"
                                "- Both records are relevant to the request.", ["d"], m).items, [])
        mixed = verify("[1] مرجعيات الذاكرة في قصص Najman Yassin stories | Authors: Mahmood, Yamama", ["d"], m)
        self.assertEqual(mixed.items[0].label, "grounded")              # [n] citation + similar title
        wrong = verify("[1] Memory references in Najman Yassin stories — Authors: Smith, John: x", ["d"], m)
        self.assertEqual(wrong.items[0].label, "misattributed")

    def test_intro_lines_are_not_items(self):
        self.assertEqual(parse_items("Based on the catalogue, I recommend the following:\n"), [])


class RealDataQuirkTests(unittest.TestCase):
    """Problems observed in the University of XXXX-4 OJS harvest (October 2026)."""

    def test_persian_letters_become_arabic(self):
        from librag.text import clean_text
        self.assertEqual(clean_text("استخلاص وتشخیص الأحماض الدهنیة لبکتریا"), "استخلاص وتشخيص الأحماض الدهنية لبكتريا")
        self.assertEqual(normalize("تأثیر"), normalize("تأثير"))

    def test_keyword_artefacts_removed(self):
        from librag.dataset import clean_subjects
        raw = ["Key words: Biota orientails,,", "root rot,,", "[]", "العدد الحالي", "Correlation", "correlation",
               "الكلمات المفتاحية: التعلم الالکترونی، الجامعات"]
        self.assertEqual(clean_subjects(raw), ["Biota orientails", "root rot", "Correlation", "التعلم الالكتروني", "الجامعات"])

    def test_language_variants_of_templates(self):
        r = Record("x", "s", {"ar": "عنوان عربي", "en": "English title"}, ["A"], ["water", "مياه"], {}, "en", 2020)
        self.assertEqual(record_text(r, ["title", "subjects"], "en"), "Title: English title | Subjects: water")
        self.assertEqual(record_text(r, ["title", "subjects"], "ar"), "Title: عنوان عربي | Subjects: مياه")
        only_ar = Record("y", "s", {"ar": "عنوان عربي فقط"}, [], [], {}, None, None)
        self.assertEqual(record_text(only_ar, ["title"], "en"), "Title: عنوان عربي فقط")   # falls back

    def test_one_failing_journal_does_not_stop_the_harvest(self):
        from unittest import mock
        from librag import loaders
        from librag.loaders import oai

        def fake_harvest(url, name, raw_dir, **kw):
            if "bad" in url:
                raise oai.OAIError("server down")
            yield Record(f"{name}:{url}", name, {"en": "A real looking title"})

        src = {"type": "oai", "name": "u", "url_template": "https://{code}.x/oai", "journals": ["good", "bad", "ok"]}
        with mock.patch.object(oai, "harvest", fake_harvest):
            recs = list(loaders.load_source(src, Path(tempfile.mkdtemp()), log=lambda *_: None))
        self.assertEqual(len(recs), 2)


def _oai_record(i: int, title: str) -> str:
    return (f"<record><header><identifier>oai:x:article/{i}</identifier><datestamp>2024-01-01</datestamp>"
            f"<setSpec>jr:Article</setSpec></header><metadata><oai_dc:dc xmlns:oai_dc=\"http://www.openarchives.org/OAI/2.0/oai_dc/\" "
            f"xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title xml:lang=\"en\">{title}</dc:title>"
            f"<dc:creator>Author, A</dc:creator><dc:date>2024</dc:date></oai_dc:dc></metadata></record>")


class _FakeResp:
    def __init__(self, status: int, body: str):
        self.status_code, self.content, self.headers = status, body.encode("utf-8"), {}

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} Server Error")


class _FakeOJS:
    """Page 1 works, page 2 always fails with HTTP 500 (one broken record), record 4 cannot be delivered."""
    OPEN = '<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'

    def __init__(self):
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        v = params.get("verb")
        if v == "ListRecords" and "resumptionToken" not in params:
            body = "".join(_oai_record(i, f"Working title number {i}") for i in (1, 2))
            return _FakeResp(200, self.OPEN + f"<ListRecords>{body}<resumptionToken>T2</resumptionToken></ListRecords></OAI-PMH>")
        if v == "ListRecords":
            return _FakeResp(500, "error")
        if v == "ListIdentifiers":
            hs = "".join(f"<header><identifier>oai:x:article/{i}</identifier><datestamp>2024</datestamp></header>" for i in (1, 2, 3, 4))
            return _FakeResp(200, self.OPEN + f"<ListIdentifiers>{hs}</ListIdentifiers></OAI-PMH>")
        if v == "GetRecord":
            i = int(params["identifier"].rsplit("/", 1)[1])
            if i == 4:
                return _FakeResp(500, "error")
            return _FakeResp(200, self.OPEN + f"<GetRecord>{_oai_record(i, f'Recovered title number {i}')}</GetRecord></OAI-PMH>")
        raise AssertionError(params)


class HarvestFallbackTests(unittest.TestCase):
    def test_failing_page_is_recovered_record_by_record(self):
        from unittest import mock
        from librag.loaders import oai

        tmp = Path(tempfile.mkdtemp())
        with mock.patch.object(oai.time, "sleep", lambda *_: None):
            recs = list(oai.harvest("https://x/oai", "s", tmp, session=_FakeOJS(), log=lambda *_: None,
                                    max_retries=2, page_dir=tmp))
        self.assertEqual(sorted(r.id for r in recs), ["s:oai:x:article/1", "s:oai:x:article/2", "s:oai:x:article/3"])
        info = (tmp / "HARVEST_INFO.txt").read_text(encoding="utf-8")
        self.assertIn("records_the_server_could_not_deliver=1", info)
        self.assertEqual((tmp / "FAILED_IDENTIFIERS.txt").read_text(encoding="utf-8"), "oai:x:article/4")

    def test_login_page_is_reported_as_closed_endpoint(self):
        from librag.loaders import oai
        with self.assertRaises(oai.OAIUnavailable):
            oai.parse_page(b"\n\n<!DOCTYPE html><html><head><title>Login</title></head></html>", "s")

    def test_front_matter_dropped(self):
        from librag.dataset import clean_record
        r = Record("x", "s", {"en": "Editorial", "ar": "الافتتاحية"})
        self.assertIsNone(clean_record(r, set(), 5))


if __name__ == "__main__":
    unittest.main()
