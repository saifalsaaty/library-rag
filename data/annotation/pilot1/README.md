# Pilot 1 — relevance judgements (October 2026)

Archived so that `04_make_annotation_pool.py` can create fresh workbooks for pilot 2.
These files are NOT part of the evaluation: pilot queries are excluded from the test set.

| File | Content |
|---|---|
| queries_pilot1.xlsx | the 10 pilot queries. Q005 and Q010 relabelled answerable 0 → 1 because the judges found relevant records |
| judgements_A1.xlsx, judgements_A2.xlsx | the two independent judgement workbooks (307 pairs each) |
| pool_manifest.csv | which systems retrieved each pooled record |
| adjudication_needed.xlsx | the 142 disagreements exported by step 5 |
| qrels_pilot1.txt, agreement_pilot1.csv, qrels_summary_pilot1.csv | step 5 output |
| calibration_pilot1.xlsx | 60 disagreements (all 40 of the 0-vs-2 cases plus 2 one-point cases per query) for the calibration session |

Judging time: about 2 hours per annotator (≈ 23 s per pair).

Agreement: Cohen's κ 0.35 (binary), 0.29 (linear weighted), 53.7% raw agreement → decision rules R1–R7 added to docs/annotation_guidelines.md.

الملفات هنا للتوثيق فقط ولا تدخل في نتائج البحث.
