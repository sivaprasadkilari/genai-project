import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer, util

from config import CHUNKS_PATH, EMBEDDING_MODEL_NAME, TOP_K_BASELINE


class BaselineRetriever:
    """
    Baseline retriever:
    - SentenceTransformer embeddings
    - Cosine similarity
    - Top-k selection
    """

    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        self.chunks_path = chunks_path
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.chunks = self._load_chunks()
        self.corpus_texts = [c["text"] for c in self.chunks]
        self.corpus_embeddings = self.model.encode(
            self.corpus_texts, convert_to_tensor=True, show_progress_bar=True
        )

    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(self, query: str, top_k: int = TOP_K_BASELINE) -> List[Dict[str, Any]]:
        query_emb = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self.corpus_embeddings)[0].cpu().numpy()
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = self.chunks[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "chunk_id": chunk.get("id", idx),
                    "paper_id": chunk.get("paper_id"),
                    "section": chunk.get("section"),
                    "text": chunk["text"],
                }
            )
        return results


def run_baseline(query: str, output_path: Path) -> None:
    retriever = BaselineRetriever()
    results = retriever.retrieve(query)
    payload = {"query": query, "top_k": len(results), "results": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
