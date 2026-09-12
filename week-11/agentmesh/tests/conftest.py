"""Test fixtures - stub env so the suite needs no real API key, no real JWKS
endpoint, and no network at all.

AUTH_MODE and TASK_STORE_BACKEND are HARD-SET, not `setdefault`. `.env.example`
ships `AUTH_MODE=dev` so the browser demo works out of the box, which means a
student who runs `cp .env.example .env` would otherwise have pydantic-settings
read `dev` straight into the suite - and dev mode compares the tests' signed
HS256 tokens against the literal string "demo-token". The JWKS path is what this
suite is here to test.
"""
from __future__ import annotations

import os

os.environ["AUTH_MODE"] = "jwks"
os.environ["TASK_STORE_BACKEND"] = "memory"
# Keep the suite hermetic: no rotating log file written to disk while testing. The
# console handler still attaches, so anything the app logs is still visible under -s.
os.environ["LOG_TO_FILE"] = "false"
# Pin the cohort mode too, for the same reason: COHORT_MODE now comes from .env, and
# a developer whose own .env says `online` must not make the cohort tests try (and
# fail) to load an online config with no tunnel URL. A real env var wins over .env.
os.environ["COHORT_MODE"] = "solo"
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("TRIAGE_MODEL", "gpt-5.4-nano-2026-03-17")
os.environ.setdefault("KNOWLEDGE_MODEL", "gpt-5.4-mini-2026-03-17")
os.environ.setdefault("JWT_ISSUER", "https://test-idp.local")
os.environ.setdefault("JWT_AUDIENCE", "agentmesh")
os.environ.setdefault("JWT_JWKS_URL", "https://test-idp.local/.well-known/jwks.json")
os.environ.setdefault("AGENTMESH_BASE_URL", "http://test")

from app.config import get_settings  # noqa: E402

# The lru_cache must not carry a .env-derived Settings into the suite.
get_settings.cache_clear()
