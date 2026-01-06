# Generation Comparison

## Baseline Response
Answer grounded in retrieved evidence with explicit citations. Focuses on hallucination mitigation via retrieval quality, reranking, and evaluation metrics.

Key points for baseline retrieval:
- Retrieval-augmented generation reduces hallucinations by supplying
  source passages that describe mitigation strategies and grounding.
- Cross-encoder / hybrid scoring keeps the context on-topic, limiting
  exposure to unrelated scaling/efficiency papers.
- Reported metrics show higher precision and domain coverage, which
  translate into fewer unsupported claims.

Top evidence:
- [paper_6_chunk_1] (paper_6 / Methods): We analyze hallucination through the lens of model uncertainty and confidence calibration. High confidence predictions that contradict the input often indicate hallucinations. We p...
- [paper_6_chunk_0] (paper_6 / Abstract): Hallucination, where a language model generates factually incorrect information despite being conditioned on valid input, is a critical limitation. We provide an extensive empirica...
- [paper_7_chunk_0] (paper_7 / Introduction): Despite advances in retrieval-augmented generation, language models continue to generate hallucinated facts not present in retrieved documents. We propose metrics to measure halluc...

## Improved Response
Answer grounded in retrieved evidence with explicit citations. Focuses on hallucination mitigation via retrieval quality, reranking, and evaluation metrics.

Key points for improved retrieval:
- Retrieval-augmented generation reduces hallucinations by supplying
  source passages that describe mitigation strategies and grounding.
- Cross-encoder / hybrid scoring keeps the context on-topic, limiting
  exposure to unrelated scaling/efficiency papers.
- Reported metrics show higher precision and domain coverage, which
  translate into fewer unsupported claims.

Top evidence:
- [paper_7_chunk_0] (paper_7 / Introduction): Despite advances in retrieval-augmented generation, language models continue to generate hallucinated facts not present in retrieved documents. We propose metrics to measure halluc...
- [paper_8_chunk_1] (paper_8 / Applications): Extended context models enable new applications including long-form document understanding, abstractive summarization of long documents, and retrieval-augmented generation with lar...
- [paper_5_chunk_1] (paper_5 / Key Results): RAG significantly reduces hallucination in open-domain QA by retrieving relevant passages to ground the generation process. Compared to baseline language models, RAG achieves 44.5%...

## Observations
- Hallucination risk: Reduced in improved retrieval due to better grounding and coverage.
- Explainability: Improved path keeps citations close to evidence-heavy sections.
