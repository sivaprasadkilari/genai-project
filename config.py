from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_BASELINE = 5
TOP_K_IMPROVED_CANDIDATES = 15
TOP_K_IMPROVED_FINAL = 5

BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

RANDOM_SEED = 42
