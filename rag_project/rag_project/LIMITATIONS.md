# LIMITATIONS - what this evaluation cannot tell you

- **n = 8.** One query = 12.5 points of accuracy. 100% vs 88% is a one-query difference; with 8 items the 95% interval on 8/8 is roughly 63-100% (Clopper-Pearson). Treat accuracy as a smoke test, not an estimate.
- **Designed after seeing the queries.** The version logic, the clause cross-reference regex, the abstention threshold (0.5) and the generic-word list were written knowing the exact eight questions and the corpus. The "resolved 8/8" is therefore a development-set score. A held-out question set (e.g. "limit in March 2023?", "domestic per-diem in 2022?", other amended clauses, an undated conflicting pair) is needed.
- **The reported run used no LLM.** Numbers in results/ come from the extractive generator and hashing embeddings (no network access when I built it). It is deterministic and cannot fabricate or negate, so `fabricated`/`negation` rates are not measured by it, and naive Q8 abstention (zero term overlap) is not what an LLM baseline would do. The LLM code path (Anthropic/OpenAI-compatible/Ollama) is written but was **not executed** by me; run it once and check results/.
- **Heuristic grader.** Regex gold checks: a correct answer phrased unexpectedly (e.g. Q3 "No - excluded") can be marked wrong, and a wrong answer containing a required figure can be marked right. Read per_query.csv.
- **Tiny corpus -> retrieval is easy.** 12-22 chunks, top-3 almost always contains the answer (hit@3 = 1.00). Retrieval quality differences at real scale are not visible here. P@3 is capped by how few chunks hold the answer.
- **Chunk-size distribution** is over 12/22 chunks; it illustrates, not characterises.
- **Regex metadata extraction** works for these phrasings only ("Effective <date>", "Supersedes the policy dated", "clause X of the <title> is"). Different wording would silently not link.
- Year-only dates in a question are mapped to 31 Dec of that year; month-year to the 15th.
