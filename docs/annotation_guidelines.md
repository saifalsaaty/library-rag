# Annotation guidelines / إرشادات التحكيم

These guidelines are part of the study protocol. Keep them unchanged during annotation and report them (or cite this file in the code repository) in the paper.

هذه الإرشادات جزء من بروتوكول الدراسة. لا تُعدَّل أثناء التحكيم، وتُذكر في البحث أو يُشار إليها في مستودع الكود.

---

## 1. Writing the test queries / كتابة استعلامات الاختبار

**Who.** The two librarians of the University of XXXX-4 Central Library who also judge relevance (A1 and A2), plus real queries from the library's search logs if the library can provide them (anonymised: query text only, no user identifiers, dates or IP addresses).

**من يكتبها.** أمينا المكتبة المركزية اللذان يحكّمان الصلة (A1 وA2)، مع استعلامات حقيقية من سجلات البحث إن أمكن، بعد إزالة كل ما يدل على المستفيد (نص الاستعلام فقط).

**How many.** Target 80 queries:

| | Arabic | English |
|---|---|---|
| answerable | 32 | 32 |
| unanswerable (topic deliberately outside the collection) | 8 | 8 |

Within each language: 16 `topical` (12 answerable + 4 unanswerable), 12 `exploratory` (8 + 4) and 12 `known_item` (all answerable), i.e. 40/30/30%. Half of each language's queries are conversational requests and half keyword queries. So that no discipline dominates, each language's 32 answerable queries are spread over the 21 journals: at least one per journal, the rest in proportion to the journal's number of records. The workbook `data/queries/queries.xlsx` comes with this plan filled in (columns `type`, `answerable`, `field`, `journal`, `phrasing`); writers may swap fields between two rows of the same language.

**العدد.** 80 استعلاماً: 40 عربياً و40 إنجليزياً، منها 16 خارج نطاق المجموعة عمداً لقياس الامتناع الصحيح.

**Query types / أنواع الاستعلام**

- `known_item` — the user is looking for a specific work they partly remember. البحث عن عمل بعينه يتذكر المستفيد جزءاً منه.
  *EXAMPLE (do not copy):* "the 2019 study on groundwater salinity in Nineveh"
- `topical` — resources about a defined subject. مصادر عن موضوع محدد.
  *EXAMPLE:* "أبحاث عن تلوث مياه نهر دجلة"
- `exploratory` — a broad or vague information need, as asked at a reference desk. حاجة واسعة أو غامضة كما تُطرح في مكتب الإرشاد.
  *EXAMPLE:* "I need background reading for a thesis on e-learning in Iraqi universities"

**Phrasing mix / صياغة الاستعلامات** (lesson from the pilot: pilot queries read like paper titles)

- About half of the queries must be **conversational requests**, as typed into a chat box or asked at the reference desk:
  "أحتاج بحوثاً عن…", "هل توجد دراسات حول…", "I'm writing a thesis on … what can I read?"
- The other half may be **short keyword queries**, as typed into an OPAC: "القوة الانفجارية لاعبي كرة اليد".
- Do not write queries that look like the title of a paper. لا تكتب الاستعلام بصيغة عنوان بحث.

نحو نصف الاستعلامات يجب أن يكون **طلباً حوارياً** كما يُكتب في نافذة محادثة أو يُسأل في مكتب الإرشاد، والنصف الآخر **كلمات مفتاحية قصيرة** كما تُكتب في الفهرس الآلي.

**known_item, precisely / الاستعلام عن عمل بعينه:** the user remembers *a specific paper* only partly: part of the title, an author, roughly when or in which journal it appeared. Example: "a study by Al-Taie around 2015 on wheat irrigation with treated water". A general subject such as "history of classical mechanics" is **topical**, not known-item.

**Unanswerable queries / الاستعلامات غير القابلة للإجابة:** choose a field that none of the 21 journals covers, such as astronomy, oceanography or aerospace engineering. Avoid neighbouring fields: journalism and social media, food packaging and similar topics already appear in the education, regional-studies and agriculture journals.

**Rules / قواعد**

1. Write queries the way users really ask, including conversational phrasing. اكتب كما يسأل المستفيد فعلاً.
2. Do not look at specific records while writing (avoid copying titles). لا تنسخ عناوين السجلات.
3. Unanswerable queries: choose topics clearly outside **all** the journals' disciplines (see the journal list). Topics that touch statistics, computing, agriculture or food science, medicine, engineering, law, education or the humanities are NOT safe. Safer areas: oceanography and marine biology, aerospace engineering, polar science. (Astronomy proved NOT safe: the Education and Science journal publishes galaxy photometry, and solar activity appears in the Science and Statistics journals.) If pooled judgements later find relevant records, step 5 flags the query and its label is corrected.
4. `known_item` queries must be written as a user *remembers* a work: a partial, paraphrased description (topic, approximate year, journal or author). Never paste the article's title into the query. A pasted title makes retrieval artificially easy and biases the comparison towards lexical search.
5. For every `known_item` query, first choose the work on the website of one of the **harvested** journals (sheet المجلات), then copy its exact title into the `target` column. The query itself stays a paraphrase. Step 4 checks that the target exists in the collection, adds it to the pool, and shows it to the annotators. A target that is not in the collection is reported and the query must be replaced.
6. Record who wrote each query (`written_by`) and its origin (`source`).
7. Step 4 checks every query automatically, at two levels:
   - six or more consecutive words of a record title → the script stops until the query is rewritten;
   - four or more content words in a row shared with a title or keyword list → a warning, recorded in `results/tables/query_copy_check.csv`. In the test set every flagged query is rephrased by its author, unless review shows that the shared phrase is a journal name, a standard term (e.g. "type 2 diabetes") or a conversational formula ("what do we know about"); such queries are listed under `copy_check_accepted` in `config.yaml` and marked in the CSV.

   A list of keywords copied from one paper is also not a real request, even if it passes the check.
8. Correct accidental spelling mistakes (e.g. "detcetion", "lightweghit"). Misspellings penalise word-matching systems such as BM25 more than embedding models, so uncontrolled typos would bias the comparison. Queries taken from real search logs are the exception: keep them exactly as typed.

يفحص السكربت 4 الاستعلامات آلياً: يتوقف إذا كرر استعلام ست كلمات متتالية أو أكثر من عنوان بحث، ويحذّر إذا اشترك في أربع كلمات متتالية مع عنوان أو كلمات مفتاحية، وعندها يعيد الكاتب الصياغة.
صحّح الأخطاء الإملائية غير المقصودة، لأنها تضر البحث بالكلمات (BM25) أكثر من النماذج الدلالية فتنحاز المقارنة. يُستثنى من ذلك ما يؤخذ من سجلات البحث الحقيقية فيبقى كما كُتب.

لكل استعلام من نوع `known_item`: اختر البحث أولاً من موقع إحدى المجلات المحصودة، وانسخ عنوانه الدقيق في عمود `target`، ثم اكتب الاستعلام بصياغة تقريبية كما يتذكره المستفيد.

**Lessons from the pilot (10 queries, October 2026)**
- Two queries labelled unanswerable (hierarchical clustering; smart food packaging) fall inside the scope of the Statistics, Computer Science and Agriculture journals.
- One known-item query reproduced an article title almost verbatim.
- The intended paper of another known-item query is not in the harvested collection, and for the third the intended work was not recorded, so the annotators had to guess it. Hence the `target` column (rule 5).
- These were corrected in the rules above before the final set was written; the pilot queries are excluded from the final evaluation.

**Lessons from pilot 2 (6 queries)**
- One keyword query was the keyword list of a single article, copied word for word; another reproduced a distinctive title phrase ("Eye Detection in Faces Images").
- One known-item query named only the journal and year, with nothing that identifies the intended work. A known-item query must contain at least one feature of the target (its topic, an author or a remembered phrase) but must not copy its title.
- One English keyword query contained two misspellings, hence rule 8.
- Step 4 now screens every query automatically. A query that shares 4 or more content words in a row with any title or keyword list is flagged (`results/tables/query_copy_check.csv`) and must be rephrased by its author.

يفحص السكربت 4 كل استعلام آلياً: إذا اشترك في 4 كلمات متتالية أو أكثر مع عنوان بحث أو قائمة كلماته المفتاحية يظهر تحذير، ويعيد كاتب الاستعلام صياغته بكلماته.

---

## 2. Relevance judgements / أحكام الصلة

Workbooks `data/annotation/judgements_A1.xlsx`, `judgements_A2.xlsx` (one per annotator).

Known-item queries are not in the workbooks: their target is the only relevant record (R5, R6) and is recorded automatically (`known_item_qrels.csv`). In pilot 2 both annotators had graded the targets 2 and all 73 other records 0, so judging these pools adds no information. The English-only MiniLM does not contribute records to the pools of Arabic queries (`pool_exclude` in `config.yaml`).

استعلامات known-item لا تظهر في ملفات التحكيم: البحث المقصود هو وحده ذو الصلة ويُسجَّل آلياً.

| Grade | Meaning | المعنى |
|---|---|---|
| 2 | Highly relevant — a librarian would recommend this record for the request | وثيق الصلة: يوصي به أمين المكتبة لهذا الطلب |
| 1 | Partially relevant — related, useful as background or for part of the request | صلة جزئية: مفيد كخلفية أو لجزء من الطلب |
| 0 | Not relevant | غير ذي صلة |

**Decision rules (added after the pilot) / قواعد القرار (أضيفت بعد التجربة الأولية)**

| Rule | Situation | Grade | الحالة |
|---|---|---|---|
| R1 | The record's **main topic** is what the request asks for | 2 | الموضوع الرئيس للسجل هو المطلوب نفسه |
| R2 | Same topic but a different population, setting or method, or it covers only part of the request (e.g. iris recognition or pupil detection for a request about eye detection) | 1 | الموضوع نفسه لكن بفئة أو سياق أو منهج مختلف، أو يغطي جزءاً من الطلب |
| R3 | Shares **words** with the request but not its meaning (e.g. "skills" in a basketball study for a request about computer-use skills; "smart" or "requirements" in an IoT paper for a request about smart libraries). This includes a word used in another sense ("detection" in intrusion or signal detection when the request is about object detection with YOLO) and look-alike words ("الانموذج الواقعي" or "المعزز بالأنشطة" for a request about الواقع المعزز, augmented reality) | 0 | يشترك مع الطلب في كلمات لا في المعنى، ومنها الكلمة بمعنى آخر («كشف التسلل» لطلب عن كشف الأجسام) والكلمة المشابهة لفظاً («الانموذج الواقعي» لطلب عن الواقع المعزز) |
| R4 | Same broad field only (e.g. any computing paper for "computer-use skills") | 0 | المجال العام نفسه فقط |
| R5 | `known_item` request: the work named in the `target` column (or its translation or another edition) | 2 | العمل المذكور في عمود target |
| R6 | `known_item` request: any other work, even on the same subject | 0 | أي عمل آخر حتى لو في الموضوع نفسه |
| R7 | No abstract and the title is too vague to tell | 0, plus `note: vague` | العنوان غامض بلا ملخص |

1. Judge relevance to the **information need**, not word overlap. Ask: *"Would I hand this item to the user who asked?"* احكم على الحاجة المعلوماتية لا على تشابه الكلمات.
2. Judge from the metadata shown (titles, subjects, abstract when present). Most records have no abstract; the title and subjects are normally enough. اعتمد على العنوانين والكلمات المفتاحية.
3. A grade of **2 should be rare** for broad requests. As a guide, most requests have only a handful of records that deserve 2. الدرجة 2 قليلة عادة.
4. Judge every row on its own. Never fill a column by dragging or copying a value down. Take a short break every 50 rows. لا تملأ العمود بالسحب أو النسخ.
5. Judge each pair independently. Do not discuss with the other annotator until both have finished. لا نقاش قبل انتهاء الطرفين.
6. Records are shuffled and the systems that found them are hidden. ترتيب السجلات عشوائي ومصدرها مخفي.
7. Language does not affect relevance: an English record can be highly relevant to an Arabic query. اللغة لا تؤثر في الصلة.
8. Write a short `note` when unsure; it is shown when disagreements are discussed. Log the total time spent (it is reported in the paper). سجّل ملاحظة عند الشك، وسجّل الوقت الكلي.

**Calibration.** Before the main annotation, the annotators are briefed on the decision rules and instructed to apply them strictly; the pilot disagreements (`data/annotation/pilot1/calibration_pilot1.xlsx`) can be used as worked examples. They then judge a fresh mini-set of queries independently. The main annotation starts only when Cohen's κ (binary) ≥ 0.60 on that set. Pilot 1 (10 queries, 307 pairs, about 2 h per annotator): κ = 0.35 (binary), 0.29 (weighted), 53.7% raw agreement. The main sources of disagreement were grading by shared words, known-item requests and lenient use of grade 2.

**المعايرة:** قبل التحكيم الرئيس يُعرَّف المحكّمان بقواعد القرار ويُطلب منهما الالتزام بها بدقة، ويمكن الاستعانة بحالات الخلاف في التجربة الأولى أمثلةً، ثم يحكّمان مجموعة صغيرة جديدة باستقلالية. لا يبدأ التحكيم الرئيس إلا إذا بلغ κ قيمة 0.60 على الأقل.

Pilot 2 (6 new queries, 220 pairs, about 1 h per annotator): κ = 0.69 (binary; 0.64 without the two known-item queries), 0.68 (weighted), 87.7% raw agreement. The threshold was met and the main annotation can start. Both annotators graded the two known-item targets 2 and every other record 0 (R5, R6). Of the 27 remaining disagreements, over half concerned neighbouring tasks (e.g. iris recognition or pupil detection for an eye-detection request), graded anywhere from 0 to 2; an example was added to R2. About a third were records that share a word used in another sense, or a look-alike word, with the request; examples were added to R3. The rest matched only the broad field (R4).

التجربة الثانية (6 استعلامات جديدة، 220 زوجاً): κ = 0.69 (الثنائي)، فتحقق الشرط ويبدأ التحكيم الرئيس.

**Monitoring during the main annotation.** After both annotators have judged about the first 20 queries, run `python scripts/05_import_judgements.py --partial`. It reports κ on the pairs judged so far and writes nothing. If binary κ is below 0.60, pause, discuss the disagreements together against rules R1–R7, then continue.

**المتابعة أثناء التحكيم الرئيس:** بعد نحو 20 استعلاماً شغّل السكربت 5 مع `--partial`. إذا نزل κ الثنائي عن 0.60 يتوقف المحكّمان ويناقشان الخلافات وفق القواعد R1–R7 ثم يكملان.

**Resolving disagreements.** Decision taken for the test set: disagreements are not adjudicated. The main judgements take the lower of the two grades (a record counts as relevant only when both annotators judged it so). Because the conclusions could depend on this choice, step 6 also ranks the systems with each annotator's judgements alone and with the higher grade, and reports Kendall's τ between the rankings (Voorhees, 2000). The optional consensus step remains available in step 5: the two annotators fill `final` in `adjudication_needed.xlsx` and save it as `adjudication.xlsx`.

**حسم الخلافات:** تقرر ألا تُعقد جلسة توافق. تُعتمد الدرجة الأدنى في كل خلاف، أي لا يُعدّ السجل ذا صلة إلا إذا اتفق المحكّمان على ذلك. ويرتّب السكربت 6 الأنظمة أيضاً بأحكام كل محكّم وحده وبالدرجة الأعلى، ويقيس ثبات الترتيب بمعامل Kendall τ.

**Double judging of a subset (optional).** If `evaluation.double_judged_fraction` in `config.yaml` is below 1 (e.g. 0.3), A1 judges every query and A2 judges a stratified random sample of whole queries (by language, type and answerability; the list is saved as `double_judged_queries.csv`). Agreement and the consensus step use the shared queries; A1's grades are used for the others. This design (single assessor plus a double-judged sample) is common in IR test collections; disagreement between assessors changes absolute scores but rarely the ranking of systems (Voorhees, 2000).

---

## 3. Human evaluation of generated answers / التقييم البشري لإجابات النماذج

Workbooks `results/generation/human_eval_A1.xlsx` (150 items) and `human_eval_A2.xlsx` (50 of the same items, for agreement). Each row is one resource mentioned in one answer, sampled within each automatic label. Model, condition and the automatic label are hidden.

How to check `in_catalogue`: the column `catalogue_search_results` lists the ten catalogue records closest to the cited title (BM25 over the full records, not the detector's title matcher); answer `y` if the cited work is one of them. `python scripts/search_catalogue.py` allows further searches if needed. For `exists_elsewhere`, search the title in Google Scholar; answer `unknown` if unsure after one search. Work through the rows in order and do not discuss items with the other evaluator.

للتحقق من وجود البحث في المجموعة انظر العمود `catalogue_search_results` الذي يعرض أقرب 10 بحوث من المجلات إلى العنوان. وللتحقق من وجوده خارج المجموعة ابحث عنه في Google Scholar.

| Column | Question | السؤال |
|---|---|---|
| in_provided_records (y/n) | Is this item one of the records listed in `provided_records`? (answer `n` if the list is empty) | هل المادة من السجلات المعطاة للنموذج؟ |
| in_catalogue (y/n) | Does the item exist in the collection? (search the collection's OPAC or journal site) | هل المادة موجودة في المجموعة؟ |
| exists_elsewhere (y/n/unknown) | Only if not in the catalogue: does the work exist at all (Google Scholar, WorldCat, publisher site)? | إن لم تكن في المجموعة: هل للعمل وجود حقيقي أصلاً؟ |
| author_correct (y/n/na) | Are the stated authors correct for that work? | هل المؤلفون المذكورون صحيحون؟ |
| description_faithful (y/partial/n) | Is what the answer says about the item supported by the record's metadata? | هل وصف النموذج للمادة تدعمه بيانات السجل؟ |

These judgements are used to (a) measure the accuracy of the automatic hallucination detector, (b) separate truly fabricated works from real works the library does not hold, and (c) estimate the faithfulness of the descriptions.

تُستخدم هذه الأحكام لقياس دقة الكاشف الآلي، وللتمييز بين المصادر المختلقة والمصادر الحقيقية غير الموجودة في المكتبة، ولتقدير أمانة الوصف.
