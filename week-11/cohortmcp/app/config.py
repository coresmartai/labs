"""Typed settings - loaded once via lru_cache, never read via os.environ in business code."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings - read from .env once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API key - required by the eval endpoint at call time (llm.py fails loud
    # on an empty key). Defaulting to "" keeps the packaged MCP server runnable in
    # a clean environment: the stdio tool loop never touches the model. ---
    openai_api_key: str = ""

    # --- Optional base URL. Unset -> the provider's default endpoint. Set it and the
    # exact same code talks to any OpenAI-compatible endpoint: a gateway, a proxy, or
    # an open-weight checkpoint you self-host behind vLLM/Ollama. The pin below is
    # then whatever *that* endpoint calls the model. ---
    openai_base_url: str | None = None

    # --- Model pin (never "latest" - silent upgrades break the eval harness) ---
    openai_model: str = "gpt-5.4-nano-2026-03-17"

    # --- App ---
    log_level: str = "INFO"

    # --- Tool description quality - flip it and restart to reproduce the eval failure mode ---
    tool_description_quality: Literal["good", "bad"] = "good"

    # --- MCP server transport (python -m app.mcp_server) ---
    mcp_server_transport: Literal["stdio", "http"] = "stdio"
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8766


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cache once - .env is read exactly once per process."""
    return Settings()
