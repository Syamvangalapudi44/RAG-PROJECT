# DECISIONS

1. **Two chunkers, genuinely different.**
   - `fixed`: 300-char windows, 60 overlap, structure-blind (can cut a table row or clause in half).
   - `structure`: parses blocks -> one chunk per numbered clause, per table row (rendered `Band=4; Role level=Lead / Manager; Notice period=90 days`), per Q&A pair, per paragraph. Every chunk is *embedded* with its document header (title, effective date, "Supersedes ...", table column names) but the answer text is the body only.
2. **Embeddings / index.** Default: my own TF-IDF-style hashing embedding (2048 dims, stemmed unigrams+bigrams) with plain NumPy cosine. Chosen so the script runs with no downloads. `EMBED=st` switches to all-MiniLM-L6-v2.
3. **Generation.** Provider-agnostic LLM call (Anthropic / OpenAI-compatible / Ollama) with a strict "cite [file]; reply ABSTAIN if unsupported" prompt. With no key, a deterministic extractive generator is used so the pipeline is always runnable and reproducible. It is a stand-in, not an LLM (see LIMITATIONS.md).
4. **Conflict policy (Step 4) = metadata at ingestion + date-aware resolution + mandatory disclosure + refusal as last resort.**
   - Ingestion extracts: effective date, "Supersedes ... dated X" links, and explicit clause cross-references ("clause 5.2 of the Equipment and Asset Policy").
   - Query time: pick the version *in force on the date asked* (date from the question, e.g. "June 2019", else today). This is what makes Q6 work: pure "newest wins" would be wrong for it.
   - The memo does not repeat the old figure (25,000) and a keyword/semantic search for the policy clause never surfaces it, so it is linked via the clause cross-reference. When the clause is retrieved, the memo is attached and the original clause text is removed from the in-force context and shown only as "superseded".
   - Never silent: every answer states which version was used and lists superseded values labelled as such.
   - If two in-force chunks of the same family cover the same clause with different figures and dates cannot order them, the system refuses and shows both (self-tested on a synthetic pair; not triggered by this corpus).
   - Rejected: "always show both and let the user choose" (pushes work to the user when a deterministic rule exists - I keep it only for the genuinely unresolvable case); "recency only" (wrong for historical questions); "refuse on any conflict" (would make Q1/Q5 unanswerable although the corpus states which document wins).
5. **Abstention guard.** Abstain if <50% of the question's topical terms (stopwords, generic words, months, numbers removed) occur in the top-3 chunks. Threshold and the generic-word list were written *after* seeing the 8 queries - see LIMITATIONS.md.
6. **Metrics.** Accuracy = automatic grading against a gold regex spec (must include / must not assert unless the line is labelled superseded / must cite a gold file). P@3 = relevant chunks in top-3 / 3 over Q1-Q7 (Q8 has nothing relevant). A chunk is relevant only if it contains the gold evidence text, so a stale chunk is *not* relevant. P@3 is structurally capped (<1) when only one or two chunks hold the answer; hit@3 is also reported.
