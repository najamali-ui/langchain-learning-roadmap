"""config.py — central configuration loaded from .env"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM (Qwen via OpenAI-compatible endpoint) ──────────────────────────────
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "ollama")
QWEN_MODEL: str      = os.getenv("QWEN_MODEL", "qwen2.5:7b")

# ── Embeddings ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ── FAISS ───────────────────────────────────────────────────────────────────
FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")

# ── Memory ──────────────────────────────────────────────────────────────────
MEMORY_BUFFER_SIZE: int = 4           # keep only last 2 exchanges to avoid bloated context

# ── Scoring ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = 5                # default candidates to surface per query
