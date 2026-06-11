"""
config.py — Centralized configuration for AI Research Assistant.
Loads environment variables and exposes typed settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM Settings (Ollama)
    # Fix 1: Removed stray spaces around colon in default model name
    # Fix 2: Changed default from qwen:latest (no tool support) to qwen2.5:7b
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    AGENT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.3"))

    # Search (SerpAPI)
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")

    # Database
    DB_PATH: str = os.getenv("DB_PATH", "research_history.db")

    # Search
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

    @classmethod
    def is_serpapi_configured(cls) -> bool:
        return bool(cls.SERPAPI_API_KEY)


config = Config()
