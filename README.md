# genai-project

A small reproducible example repo for experimenting with RAG-style retrieval over a small set of LLM research papers.

Repository layout:
- data/
  - papers/        : store the original 10 PDF research papers (not included here)
  - raw_text/      : optionally extracted plain text
  - chunks.json    : JSON list of chunked document objects
- baseline/
  - baseline_retrieval.py
  - baseline_results.json
- improved/
  - improved_retrieval.py
  - improved_results.json
- evaluation/
  - comparison_table.md
  - evaluation_summary.md
- main.py
- config.py
- requirements.txt

Usage:
1. Put your PDFs into `data/papers/`.
2. Run `python main.py` to run both baseline and improved retrieval pipelines and produce results in their respective folders.
3. Review `evaluation/` to compare before & after.

License: Add a license if desired.
