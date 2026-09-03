"""Typed Settings via pydantic-settings. Loaded once via @lru_cache."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings - read from .env once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI API
    openai_api_key: str

    # Pinned model versions - no "latest" anywhere
    llm_model: str = "gpt-5.4-mini-2026-03-17"
    embed_model: str = "text-embedding-3-large"   # same as Week 4 KnowledgeVault

    # Qdrant - the same store Week 4 KnowledgeVault wrote.
    # local  (default) = embedded in this process, reading QDRANT_LOCAL_PATH.
    #                    Point it at (or copy) Week 4's qdrant_local/ folder.
    #                    One process at a time - stop the Week 4 server first.
    # server           = a running Qdrant at QDRANT_URL (Docker or Cloud).
    qdrant_mode: str = "local"
    qdrant_local_path: str = "./qdrant_local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledgevault"

    # Retrieval thresholds - calibrated for text-embedding-3-large cosine scores
    # (top1_dense is used for the gate, not the RRF score).
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.55
    spread_delta: float = 0.08

    # Paths
    golden_dataset_path: str = "./data/golden_dataset.json"

    # Logging
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cache the settings object so we read .env exactly once."""
    return Settings()
