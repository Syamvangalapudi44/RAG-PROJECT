# WRITEUP

## 0. Setup used for the numbers below  (EDIT THIS before submitting)
- Embeddings: built-in hashing TF-IDF (2048-d), NumPy cosine. Generation: deterministic extractive generator (**no LLM**) for the committed run.
- LLM used for the AI-assisted parts / final run: **<<FILL IN: model name, e.g. Claude / GPT / Llama 3 via Ollama>>**; tools: **<<FILL IN: VS Code / Cursor / Claude Code / claude.ai ...>>**.
- If you re-run with a key (`python run.py` auto-detects it) paste the new table here and update §1-§4: the numbers below will change.

## 1. Which strategy I would deploy
Structure-aware chunking with the conflict-aware ("resolved") pipeline.

| chunking | pipeline | accuracy | P@3 | hit@3 | chunks | mean chars | abstention |
|---|---|---|---|---|---|---|---|
| fixed | naive | 5/8 | 0.38 | 1.00 | 12 | 222 | 12% |
| fixed | resolved | 7/8 | 0.29 | 0.86 | 12 | 222 | 12% |
| structure | naive | 5/8 | 0.43 | 1.00 | 22 | 162 | 12% |
| structure | resolved | 8/8 | 0.38 | 1.00 | 22 | 162 | 12% |

(abstention 12% = 1/8, the correct refusal on Q8. Plot: results/chunk_size_distribution.png.)
Reasons: the fixed splitter broke the notice table (Q4 fragment) both before and after the fix, while structure chunks kept each table row with its column names; structure had higher P@3 (0.43 vs 0.38 naive) and never lost a hit. Fixed chunks are fewer and larger (222 vs 162 chars) but sizes are irregular (some are fragments). Conflict handling, not chunking, fixed the three-to-four stale/conflict failures (naive 5/8 -> 8/8 with structure chunks), so the two choices are complementary. Weight on these figures: see §4.

## 2. Conflict-resolution design and why
Ingestion extracts effective date, "supersedes" links and clause cross-references; at query time the system selects the document version in force on the date in the question (default: today), applies amendments whose effective date has passed, strips the amended clause from the in-force context, and always prints which version was used plus the superseded values labelled as such. If sources conflict and dates cannot order them, it refuses and shows both.
Why not the alternatives: recency-only breaks Q6 (June 2019 must use the 2019 policy); "let the user choose" is only right when the corpus gives no ordering; "refuse on conflict" would make Q1/Q5 unanswerable although the corpus itself says 2024 supersedes 2019. The memo is the hard case: it does not restate 25,000, so similarity search for the clause never finds it; the clause cross-reference is what links them.
Before/after: naive 5/8 -> resolved 7/8 (fixed) and 8/8 (structure). Q1, Q5 (stale 2019 figures), Q2 (structure; original 25,000 clause) were fixed.

## 3. Queries still wrong after the fix
- fixed + resolved, Q4 (fragment): the table was split by the window. With more time: parent-document expansion or table-aware splitting for any strategy shipped; do not deploy fixed-size on tables.
- None for structure + resolved in this run - but see §4: that is a development-set score, and an LLM-based run may expose negation (Q3) or fabrication (Q8) failures that the extractive generator cannot produce.

## 4. What the evaluation cannot tell you
Eight queries means one question = 12.5 points; 8/8 has a 95% lower bound near 63%. I wrote the logic, thresholds and gold regexes knowing these exact queries, so 8/8 is closer to "the design handles the known traps" than to an expected production accuracy. The committed run used no LLM, so hallucination and negation behaviour is unmeasured. Retrieval is trivially easy at 12-22 chunks. Full list: LIMITATIONS.md.

## 5. Where AI was used / where I did not accept its output  (WRITE THIS YOURSELF - be truthful)
- AI assistant used for: **<<FILL IN honestly: e.g. scaffolding run.py, regex drafting, drafting these documents>>**.
- One place I did not accept the output: **<<FILL IN a real one. Examples of things worth checking and possibly rejecting: the grader accepting an answer that quotes an old figure; the "naive abstains on Q8" result being an artefact of the extractive generator; the abstention threshold being tuned to the queries; the claim of 100% accuracy.>>**
