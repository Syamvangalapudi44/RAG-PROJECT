# RAG over a contradictory corpus

    pip install -r requirements.txt
    python run.py                      # works with zero keys (extractive generator, hash embeddings)

Use a real LLM / embeddings (recommended before submitting - see WRITEUP.md §0):

    export ANTHROPIC_API_KEY=...       # or OPENAI_API_KEY=...  (OPENAI_BASE_URL for any compatible proxy)
    # or: LLM_PROVIDER=ollama LLM_MODEL=llama3
    # optional: pip install sentence-transformers && export EMBED=st
    python run.py

Outputs go to `results/`: comparison.md (summary + per-query table), per_query.csv, summary.json,
chunk_size_distribution.png.

Files: corpus/ (6 docs, untouched) · run.py · DECISIONS.md · ERROR_ANALYSIS.md · LIMITATIONS.md · WRITEUP.md
