"""
Configuration for genai-project.
Adjust paths and model names as needed.
"""
from pathlib import Path

ROOT = Path(__file__).parent

DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
RAW_TEXT_DIR = DATA_DIR / "raw_text"
CHUNKS_FILE = DATA_DIR / "chunks.json"

BASELINE_DIR = ROOT / "baseline"
IMPROVED_DIR = ROOT / "improved"
EVALUATION_DIR = ROOT / "evaluation"

# Embedding / model config
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers
TOP_K = 5

# OpenAI (if used in improved pipeline)
OPENAI_API_KEY = None  # set via environment variable or override at runtime
