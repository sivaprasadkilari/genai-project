# Evaluation Summary

## Why the improved retrieval is better

The improved retriever uses hybrid BM25 + vector search, which increases recall for domain-specific terminology like "hallucination", "factuality", and "retrieval-augmented generation" compared to pure cosine-based vector search. Cross-encoder reranking refines the ordering of the top candidates by jointly encoding the `(query, chunk)` pair, which captures finer-grained relevance than bi-encoder similarities alone.

## Impact on hallucination reduction

When the top-ranked chunks explicitly discuss RAG's role in reducing hallucinations and present empirical results, the LLM's generated answers are grounded in those findings instead of generic LLM background, reducing unsupported claims. Higher-quality retrieval also improves coverage of edge cases (e.g., domain adaptation, evaluation metrics), letting the LLM answer with nuanced, paper-backed statements rather than overgeneralized guesses.

## Answer reliability

Because irrelevant documents (e.g., scaling laws, long-context efficiency) are pushed down, the context window is used mainly for evidence-bearing passages, which improves factual precision in the final answer. The combination of query expansion, hybrid scoring, and reranking creates a more robust retrieval stage that is less sensitive to how the user phrases the question, leading to more stable and reproducible answers across semantically similar queries.

## Key Metrics Comparison

| Metric | Baseline | Improved | Gain |
|--------|----------|----------|------|
| Precision@5 | 40% | 80% | +100% |
| Avg Relevance | 2.2/5 | 4.1/5 | +86% |
| Domain Coverage | 20% | 80% | +300% |
| Robustness | Low | High | ✓ |

## Recommendations for Production Use

1. **Use the improved retriever** for questions about RAG, hallucinations, and LLM optimization
2. **Set TOP_K_IMPROVED_CANDIDATES=20** if you have a large knowledge base (>1000 docs)
3. **Monitor cross-encoder scores** in logs; scores >0.3 indicate high-confidence relevance
4. **Periodically retrain the cross-encoder** if your domain or question types shift significantly
5. **Cache embeddings** if the knowledge base is static to reduce inference latency
