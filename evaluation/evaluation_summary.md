# Evaluation Summary

This document describes how to evaluate the baseline vs improved retrieval setups.

Steps:
1. Prepare a set of evaluation queries and gold-standard relevant chunks.
2. Run `python main.py` to generate `baseline/baseline_results.json` and `improved/improved_results.json`.
3. Compute metrics such as Precision@K, Recall@K and MRR.
4. Document findings in `comparison_table.md`.

Suggestions for improvement:
- Add a cross-encoder re-ranker for higher precision.
- Add BM25 or other sparse retrieval to capture lexical matches.
- Use query expansion via an LLM for better recall.

Ensure directories are created as needed. Set repository visibility to private. Commit all files to the default branch (main).
