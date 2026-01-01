"""
Baseline retrieval pipeline:
- Loads chunked documents from data/chunks.json
- Builds embeddings with sentence-transformers
- Performs simple nearest-neighbor search (FAISS)
- Returns top-k docs per query (example queries are included)
"""
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from config import CHUNKS_FILE, EMBEDDING_MODEL, TOP_K

MODEL = SentenceTransformer(EMBEDDING_MODEL)

def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def build_index(texts: List[str]):
    embeddings = MODEL.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings

def run_baseline():
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    if not texts:
        return {"error": "no chunks found in data/chunks.json"}
    index, embeddings = build_index(texts)

    # example queries
    queries = [
        "What are the main architectural changes in recent LLMs?",
        "How to evaluate retrieval-augmented generation?"
    ]
    results = []
    for q in queries:
        q_emb = MODEL.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, TOP_K)
        retrieved = []
        for score, idx in zip(D[0], I[0]):
            retrieved.append({"score": float(score), "chunk": chunks[int(idx)]})
        results.append({"query": q, "retrieved": retrieved})
    return results

if __name__ == "__main__":
    import json
    out = run_baseline()
    print(json.dumps(out, indent=2))
