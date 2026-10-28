# Pilot 2 — relevance judgements (October 2026)

Judged with the decision rules R1–R7 added after pilot 1; the annotators were briefed on the rules and instructed to apply them strictly.
Archived so that `04_make_annotation_pool.py` can create the workbooks for the test set.
These files are NOT part of the evaluation.

| File | Content |
|---|---|
| queries_pilot2.xlsx | the 6 pilot-2 queries (3 Arabic, 3 English; 2 known-item with `target`) |
| judgements_A1.xlsx, judgements_A2.xlsx | the two independent judgement workbooks (220 pairs each) |
| pool_manifest.csv | which systems retrieved each pooled record |
| adjudication_needed.xlsx | the 27 disagreements exported by step 5 |
| qrels_pilot2.txt, agreement_pilot2.csv, qrels_summary_pilot2.csv | step 5 output |
| query_copy_check_pilot2.csv | step 4 copied-phrase screen |

Judging time: about 1 hour per annotator (≈ 16 s per pair).

Agreement: Cohen's κ 0.69 (binary), 0.68 (linear weighted), 87.7% raw agreement;
binary κ 0.64 without the two known-item queries. Both known-item targets were graded 2 by both annotators.
The pre-set threshold (binary κ ≥ 0.60) was met.

الملفات هنا للتوثيق فقط ولا تدخل في نتائج البحث.
