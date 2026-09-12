"""Typed configuration via pydantic-settings.

One settings object, loaded once via @lru_cache, injected everywhere.
No os.environ reads in business code.

Dual-model pin (per CODE_GUIDE default-models table for multi-agent weeks):
  - triage_model    - cheap routing/classification (nano)
  - knowledge_model - knowledge synthesis + action proposing (mini)

Both pins are dated strings, NEVER `-latest`. /health reports both, routed
through app/llm.py's `model_for()` so the health chip and the SDK seam can
never drift apart.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Provider key - REQUIRED. No default: a missing key is a startup error naming
    # the field, not a 500 halfway through a task.
    openai_api_key: str
    # Optional. Unset -> the provider default endpoint. Set it and the same code
    # talks to any OpenAI-compatible endpoint (gateway, proxy, self-hosted checkpoint).
    openai_base_url: str | None = None

    # Dual model pin - dated strings, NEVER `-latest`
    triage_model: str = "gpt-5.4-nano-2026-03-17"
    knowledge_model: str = "gpt-5.4-mini-2026-03-17"

    # --- Student identity ---------------------------------------------------
    # Who this agent belongs to. In a cohort session these are the fields your
    # classmates actually see: they land in the Agent Card, in the `whoami` MCP
    # tool, and in the `whoami` A2A skill. Set them before you publish your card.
    student_id: str = "stu_000"
    student_name: str = "Unnamed Student"
    student_specialty: str = "incident-triage"

    # A2A service.
    # agentmesh_base_url is stamped into the Agent Card AND into the stream_url of
    # every 202 ack, so a remote caller can only reach you if this is your PUBLIC
    # url. Behind a tunnel, set it to the tunnel url - see SESSION.md.
    agentmesh_host: str = "0.0.0.0"
    agentmesh_port: int = 8000
    agentmesh_base_url: str = "http://localhost:8000"

    # Optional: URL of the cohort's published card index (a static JSON file on
    # GitHub Pages or similar). When set, `cohort_lookup` resolves classmates from
    # the live cohort instead of the built-in fixtures.
    cohort_registry_url: str | None = None

    # Auth mode - "jwks" verifies tokens against the identity provider's JWKS.
    # "dev" accepts the fixed dev bearer token below (local demos / cohort dry runs).
    # The code default is the locked one; `.env.example` ships `dev` so the browser
    # demo runs out of the box.
    auth_mode: Literal["jwks", "dev"] = "jwks"
    dev_bearer_token: str = "demo-token"
    # Dev mode grants THIS scope set - a fixed list, not "whatever the route asked
    # for". Otherwise the 403 path could never fire in dev mode and the scope check
    # would be theatre.
    dev_scopes: list[str] = ["triage:invoke", "mcp:invoke"]

    # JWT verification (used when auth_mode="jwks")
    jwt_issuer: str
    jwt_audience: str = "agentmesh"
    jwt_jwks_url: str
    jwt_jwks_cache_seconds: int = 86400  # 24 h
    jwt_leeway_seconds: int = 30

    # A2A protocol version - emitted as the `A2A-Version` response header on every
    # response. Emitted, never negotiated: a client's version header is ignored.
    a2a_protocol_version: str = "1.0"

    # MCP cohort server.
    # Standalone process: `python -m app.mcp_server` (stdio, or its own HTTP port).
    mcp_server_transport: Literal["stdio", "http"] = "stdio"
    mcp_server_port: int = 8766

    # Selects between the two description tables in `app/tools.py`. What it changes
    # is what a *classmate's* model reads when it connects to your server - the
    # description is the prompt. Resolved once at import.
    tool_description_quality: Literal["good", "bad"] = "good"

    # Co-hosting: also serve the MCP Streamable HTTP surface from the A2A app, on
    # the SAME port, at mcp_mount_path. That is what lets one tunnel expose both
    # protocols. The mounted surface is bearer-gated by the same auth as A2A -
    # a FastAPI mount bypasses route dependencies, so main.py guards it in
    # middleware instead. stdio stays available either way for local hosts.
    cohost_mcp: bool = True
    mcp_mount_path: str = "/mcp"
    # Scope a caller needs for the co-hosted MCP surface.
    mcp_required_scope: str = "mcp:invoke"

    # Task store. `memory` is the default and what the demo runs on; `sqlite` is the
    # same four-method interface with durable events and unbounded replay.
    task_store_backend: Literal["memory", "sqlite"] = "memory"
    task_store_sqlite_path: str = "./data/tasks.db"
    task_replay_buffer_size: int = 100

    # Logging. Console always; a rotating file when log_to_file (the default) so a
    # session leaves a record on disk with no button to press. logs/ is gitignored.
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/agentmesh.log"
    log_max_bytes: int = 1_000_000   # ~1 MB per file before it rolls over
    log_backup_count: int = 5        # keep this many rolled-over files


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Single source of truth for configuration. Cached for process lifetime."""
    return Settings()
