"""
Settings loader.

Loads environment variables from `.env` (via pydantic-settings) and exposes
them as a typed `Settings` object. Centralized + validated on startup so that
a missing key fails loudly at module-import, not silently at request time.

Why this matters: starting the server without OPENAI_API_KEY makes
pydantic-settings raise ValidationError at module import - the process exits
before accepting a connection. Loud-fail-at-boot is the production-grade pattern.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI - vision LLM describer + text embeddings
    openai_api_key: str
    # Primary vision describer: quality + speed balance
    openai_vision_model: str = "gpt-5.4-mini-2026-03-17"
    # Fast/cheap fallback for low-confidence re-describe passes
    openai_vision_fast_model: str = "gpt-5.4-nano-2026-03-17"
    openai_embed_model: str = "text-embedding-3-large"

    # Qdrant - vector store.
    # qdrant_mode="local" runs Qdrant embedded inside this process, storing data
    # under qdrant_local_path. No Docker, no server, no signup: the default for
    # the course. qdrant_mode="server" talks to a running Qdrant (Docker or
    # Qdrant Cloud) at qdrant_url. Same code, same collection, one switch.
    qdrant_mode: str = "local"
    qdrant_local_path: str = "./qdrant_local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledgevault"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    data_dir: str = "./data/sample"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached so we don't re-read .env on every request."""
    return Settings()  # type: ignore[call-arg]
