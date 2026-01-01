import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Load config values
DATA_PATH = Path("data/chunks.json")
RESULT_PATH = Path("baseline/baseline_results.json")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_chunks():
    """Load document chunks from JSON file"""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_chunks(query, chunks, top_k=5):
    """Retrieve top-k relevant chunks using cosine similarity"""
    texts = [chunk["text"] for chunk in chunks]

    query_embedding = model.encode([query])
    chunk_embeddings = model.encode(texts)

    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        results.append({
            "text": texts[idx],
            "score": float(similarities[idx])
        })

    return results


def run_baseline(query):
    chunks = load_chunks()
    retrieved = retrieve_chunks(query, chunks)

    output = {
        "query": query,
        "retrieved_chunks": retrieved
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("✅ Baseline retrieval completed.")
    print(f"Results saved to {RESULT_PATH}")


if __name__ == "__main__":
    sample_query = "What are the key challenges in scaling large language models?"
    run_baseline(sample_query)
