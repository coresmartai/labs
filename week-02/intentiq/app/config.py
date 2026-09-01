"""Typed settings loaded from .env via pydantic-settings, cached for the process lifetime."""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenAI (shared API key; covers both openai and nano providers) ---
    openai_api_key: str | None = None
    openai_model: str = "gpt-5.4-mini-2026-03-17"
    nano_model:   str = "gpt-5.4-nano-2026-03-17"

    # --- Ollama (local server - no API key needed) ---
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model:    str = "qwen3:0.6b"

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings instance.

    Reads .env once on first call; every subsequent call returns the same
    object from the lru_cache.  To pick up .env changes you must restart
    the server process - the cache is cleared on restart.
    """
    return Settings()
