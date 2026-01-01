import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from config import (
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_MODEL_NAME,
    TOP_K_IMPROVED_CANDIDATES,
    TOP_K_IMPROVED_FINAL,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
)


class ImprovedRetriever:
    """
    Improved retriever:
    - Query expansion
    - Hybrid BM25 + dense retrieval
    - Cross-encoder reranking
    """

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks_path = chunks_path
        self.bi_encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
        self.chunks = self._load_chunks()
        self.corpus_texts = [c["text"] for c in self.chunks]

        # Precompute dense embeddings
        self.corpus_embeddings = self.bi_encoder.encode(
            self.corpus_texts, convert_to_tensor=True, show_progress_bar=True
        )

        # Build BM25 index
        tokenized = [self._tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def _expand_query(self, query: str) -> str:
        """
        Lightweight expansion tailored to LLM / RAG questions.
        """
        parts = [query]
        q = query.lower()

        if "hallucination" in q or "factual" in q:
            parts.append("factuality grounding hallucination reduction retrieval augmented generation")
        if "rag" in q or "retrieval" in q:
            parts.append("retrieval augmented generation bm25 vector search hybrid search reranking")
        if "context length" in q or "long context" in q:
            parts.append("long context window attention memory compression")
        if "latency" in q or "efficiency" in q:
            parts.append("efficient inference distillation quantization")

        return " ".join(parts)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        expanded_query = self._expand_query(query)

        # BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(self._tokenize(expanded_query)))

        # Dense scores
        q_emb = self.bi_encoder.encode(expanded_query, convert_to_tensor=True)
        dense_scores = util.cos_sim(q_emb, self.corpus_embeddings)[0].cpu().numpy()

        # Normalize to [0,1]
        def norm(x: np.ndarray) -> np.ndarray:
            if x.max() == x.min():
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min())

        bm25_n = norm(bm25_scores)
        dense_n = norm(dense_scores)
        hybrid = BM25_WEIGHT * bm25_n + VECTOR_WEIGHT * dense_n

        # Candidate set
        cand_idx = np.argsort(-hybrid)[:TOP_K_IMPROVED_CANDIDATES]

        # Cross-encoder rerank
        pairs = [(query, self.corpus_texts[i]) for i in cand_idx]
        rerank_scores = np.array(self.cross_encoder.predict(pairs))
        order = np.argsort(-rerank_scores)[:TOP_K_IMPROVED_FINAL]
        final_idx = [cand_idx[i] for i in order]

        results = []
        for rank, idx in enumerate(final_idx, start=1):
            chunk = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "hybrid_score": float(hybrid[idx]),
                    "cross_encoder_score": float(rerank_scores[order[rank - 1]]),
                    "chunk_id": chunk.get("id", idx),
                    "paper_id": chunk.get("paper_id"),
                    "section": chunk.get("section"),
                    "text": chunk["text"],
                }
            )
        return results


def run_improved(query: str, output_path: Path) -> None:
    retriever = ImprovedRetriever()
    results = retriever.retrieve(query)
    payload = {"query": query, "top_k": len(results), "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
