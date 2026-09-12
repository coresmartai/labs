"""Typed application settings via pydantic-settings.

All env access goes through Settings - never read os.environ directly in
business code. Cached once via @lru_cache.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str

    # PINNED model versions - never use *-latest in production code.
    triage_model: str = "gpt-5.4-nano-2026-03-17"
    knowledge_model: str = "gpt-5.4-mini-2026-03-17"
    # Its own pin, so re-pointing the knowledge model cannot silently re-point
    # the action model too. Two models, two cost profiles, three settings.
    action_model: str = "gpt-5.4-mini-2026-03-17"

    # Hard cap on SUPER-STEPS per run, not on node executions: two nodes running
    # in one tick count once, and the two numbers diverge the moment you fan out.
    # LangGraph's default is 1000 from v1.0.6 onward, which on a graph this size
    # is no guardrail at all. Set it low, explicitly. For a
    # simple routing graph like this one, more than 10 means something is wrong.
    recursion_limit: int = 10

    # How long an approval stays good, in seconds. The gate pauses for minutes
    # and sometimes hours, and the world does not hold still while it waits.
    # Past this window the execute node re-proposes rather than acting on a plan
    # the world may have invalidated.
    approval_ttl_seconds: int = 900

    # Who may approve a gated action. Role, not ownership: the point of a human
    # gate is that somebody other than the requester decides. Comma-separated.
    approver_ids: str = "ops_lead"

    # Browsers that may call this API. NOT "*": this server exposes
    # DELETE /user/{uid} and POST /approve, and a wildcard hands both to any
    # page the user happens to have open.
    allowed_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    postgres_dsn: str = "postgresql://triage:triage@localhost:5432/triageflow"
    redis_url: str = "redis://localhost:6379/0"

    # Which vector-store backend external + episodic memory run on.
    #   "memory"   - in-process cosine index. Default. No server, no network.
    #   "pgvector" - real PostgreSQL 16 + the `vector` extension, started from a
    #                pip-installed embedded binary (pgserver). No Docker.
    # Both are real vector search behind the same interface; only the storage
    # changes. Install the optional dependency first: pip install pgserver
    memory_backend: Literal["memory", "pgvector"] = "memory"
    pgvector_data_dir: str = ".pgdata"

    langsmith_api_key: str = ""
    langsmith_project: str = "triageflow"

    app_env: str = "dev"
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Single source of truth for settings, cached for the lifetime of the process."""
    return Settings()  # type: ignore[call-arg]
