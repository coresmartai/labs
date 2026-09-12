"""Typed settings. The only module that reads the environment."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = "set-me-in-dot-env"

    # PINNED model versions - never use *-latest in production code.
    # One pin per agent, so re-pointing one cannot silently re-point another.
    triage_model: str = "gpt-5.4-nano-2026-03-17"
    access_model: str = "gpt-5.4-mini-2026-03-17"
    software_model: str = "gpt-5.4-mini-2026-03-17"
    hardware_model: str = "gpt-5.4-nano-2026-03-17"

    # Hard cap on SUPER-STEPS per run, not on node executions: two nodes running
    # in one tick count once. LangGraph's default is 1000 from v1.0.6 onward,
    # which on a graph this small is no guardrail at all. Set it low, explicitly.
    recursion_limit: int = 10

    request_timeout_seconds: int = 60
    max_completion_tokens: int = 1024
    log_level: str = "INFO"

    # Retrieval. k is deliberately small so a broken scope filter is visible.
    retrieval_k: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
