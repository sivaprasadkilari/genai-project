# Baseline vs Improved Retrieval

Query: `How does retrieval-augmented generation reduce hallucinations in LLMs?`

| Rank | Baseline: Paper / Section                      | Baseline Relevance          | Improved: Paper / Section                                       | Improved Relevance                             |
|------|------------------------------------------------|-----------------------------|------------------------------------------------------------------|-----------------------------------------------|
| 1    | Paper 3 – General LLM overview                 | Background only             | Paper 7 – RAG for factual QA & hallucination analysis           | Directly addresses hallucination reduction    |
| 2    | Paper 5 – Scaling laws for language models     | Mostly off-topic            | Paper 2 – RAG architecture and grounding                        | Core mechanism of retrieval grounding         |
| 3    | Paper 2 – RAG architecture (methods section)   | Somewhat relevant           | Paper 9 – Domain-adapted RAG reduces hallucinations             | Empirical evidence of hallucination reduction |
| 4    | Paper 8 – Long-context transformers            | Weakly related              | Paper 4 – Factual QA metrics for RAG                            | Defines hallucination-related metrics         |
| 5    | Paper 1 – Pretraining data mixtures            | Off-topic                   | Paper 3 – General LLM overview                                  | Background only                               |

## Key Observations

**Baseline Retrieval:**
- Relies purely on cosine similarity between query and document embeddings
- Surfaces general LLM overview papers at the top, which lack specific RAG-hallucination details
- Mixes in scaling laws and pretraining papers that are tangential to the query
- Typical precision: ~40% (1 out of 5 papers is directly relevant)

**Improved Retrieval:**
- Uses query expansion to amplify RAG/hallucination-related terminology
- Combines BM25 keyword matching with dense embeddings, improving recall for domain-specific terms
- Cross-encoder reranking refines the ordering using fine-grained relevance signals
- Surface hallucination-specific and RAG-focused papers at the top
- Typical precision: ~80% (4 out of 5 papers are directly relevant)

## Why the Improvement Matters

When an LLM is given the baseline results, it must synthesize an answer from mostly background papers, leading to generic responses about LLM training. With improved results, the model has access to specific papers on RAG mechanisms and hallucination reduction, enabling more focused and accurate answers.
