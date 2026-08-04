"""
Settings loader.

Loads environment variables from `.env` (via pydantic-settings) and exposes
them as a typed `Settings` object.

Why a settings object? Because reading env vars by hand from random places
in the code is how secrets end up in logs. Centralize, validate, type.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini-2026-03-17"
    openai_fallback_model: str = "gpt-5.4-nano-2026-03-17"

    # Email (Gmail SMTP)
    smtp_sender: str
    smtp_password: str

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached so we don't re-read .env on every request."""
    return Settings()  # type: ignore[call-arg]
