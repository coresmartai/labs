"""Typed settings - loaded once via @lru_cache, no os.environ in business code."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=())

    # PIN model versions - never use -latest in production.
    openai_api_key: str
    model_name: str = "gpt-5.4-mini-2026-03-17"   # "openai" provider - full reasoning
    fast_model_name: str = "gpt-5.4-nano-2026-03-17"          # "nano" provider  - small & fast

    # Retry policy (tool-level, tenacity)
    tool_retry_max_attempts: int = 3
    tool_retry_base_ms: int = 500
    tool_retry_max_ms: int = 8000

    # Schema-level retry-with-correction (Pydantic validation errors on tool args)
    schema_correction_max_attempts: int = 2

    # Loop cap - seed of the bounded agent loop (W9 OpsAssist formalises this)
    tool_loop_max_iterations: int = 5

    # Telemetry
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
