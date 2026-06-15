"""Application configuration (Singleton).

Reads from environment with the fixed-spine defaults from RAG_PIPELINE_TEMPLATE.md.
Import ``settings`` everywhere; it is constructed once.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent          # .../backend
PROJECT_ROOT = BACKEND_DIR.parent                              # .../Boodyyyy
RESEARCH_ROOT = PROJECT_ROOT / "src"                          # research repo root (has src/ package)


def _b(v: str | None, default: bool) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Process-wide configuration singleton."""

    def __init__(self) -> None:
        # --- paths ---
        self.models_dir = Path(os.getenv("MODELS_DIR", str(RESEARCH_ROOT / "artifacts")))
        self.research_root = Path(os.getenv("RESEARCH_ROOT", str(RESEARCH_ROOT)))
        self.registry_path = Path(
            os.getenv("MODEL_REGISTRY", str(BACKEND_DIR / "app" / "model_registry.json"))
        )
        self.metadata_dir = self.research_root / "data" / "metadata"

        # --- vector db (Qdrant) ---
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "") or None
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "market_news")

        # --- embeddings (e5) ---
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        self.embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))

        # --- generation (Groq) ---
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        # --- retrieval ---
        self.retrieval_mode = os.getenv("RETRIEVAL_MODE", "hybrid")  # hybrid|dense|sparse
        self.hybrid_alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
        self.top_k = int(os.getenv("TOP_K", "20"))
        self.final_k = int(os.getenv("FINAL_K", "5"))

        # --- news provider ---
        self.news_provider = os.getenv("NEWS_PROVIDER", "yfinance")  # yfinance|finnhub|newsapi|marketaux
        self.news_api_key = os.getenv("NEWS_API_KEY", "")
        self.news_lookback_days = int(os.getenv("NEWS_LOOKBACK_DAYS", "14"))

        # --- sentiment ---
        self.finbert_model = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
        self.sbert_model = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        # --- market data ---
        self.index_symbol = os.getenv("INDEX_SYMBOL", "^GSPC")
        self.universe = os.getenv("UNIVERSE", "nasdaq100")  # nasdaq100|sp500
        self.universe_max = int(os.getenv("UNIVERSE_MAX", "100"))
        self.price_lookback_days = int(os.getenv("PRICE_LOOKBACK_DAYS", "800"))

        # --- runtime ---
        self.torch_device = os.getenv("TORCH_DEVICE", "cpu")
        self.model_cache_size = int(os.getenv("MODEL_CACHE_SIZE", "64"))
        self.enable_sentiment = _b(os.getenv("ENABLE_SENTIMENT"), True)
        self.cors_origins = os.getenv("CORS_ORIGINS", "*")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
