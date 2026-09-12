"""Test fixtures - stub env so the test suite needs no real API key and no
real network access.

The three env vars below are HARD-SET, not `setdefault`: a student who follows
the README's own "flip TOOL_DESCRIPTION_QUALITY to bad and restart" exercise
leaves `bad` in their `.env`, and pydantic-settings would read it straight into
the suite. The tests assert `good`; the suite must be independent of whatever
`.env` happens to say.
"""
from __future__ import annotations

import os

# Stub env BEFORE app import. Pydantic-settings reads at module load.
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["TOOL_DESCRIPTION_QUALITY"] = "good"
os.environ["MCP_SERVER_TRANSPORT"] = "stdio"

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()
