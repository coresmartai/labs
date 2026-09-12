"""One source of truth for configuration. Every knob this project has is here."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # The peer we bridge to. In solo mode, a second AgentMesh on 8001.
    peer_base_url: str = "http://localhost:8001"
    peer_token: str = "demo-token"

    # This server. stdio is the default because it has no port and no auth
    # surface to get wrong; switch to http only when something on another
    # machine has to reach you.
    mcp_server_transport: Literal["stdio", "http"] = "stdio"
    mcp_server_port: int = 8770

    # The toggle that makes the description claim falsifiable inside your own
    # terminal rather than something you were told on a slide.
    tool_description_quality: Literal["good", "bad"] = "good"

    # TASK 4's decision, made configurable so the design memo can compare two
    # implementations rather than one implementation and one hypothesis.
    human_gate_mode: Literal["resume", "elicit"] = "resume"

    # A peer that pauses for a human can legitimately hold a task for a long
    # time. This bounds how long WE wait before handing control back.
    follow_timeout_seconds: float = 90.0

    # Only the tool-pick eval dials a model. Tests never do.
    openai_api_key: str = "unused-in-tests"
    picker_model: str = "gpt-5.4-nano-2026-03-17"


@lru_cache
def get_settings() -> Settings:
    return Settings()
