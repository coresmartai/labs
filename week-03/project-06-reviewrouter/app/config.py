"""Settings, read once at import and cached. Never re-read per request."""
import os
from functools import lru_cache

from pydantic import BaseModel


class Settings(BaseModel):
    openai_api_key: str = ""
    model: str = "gpt-5.4-mini-2026-03-17"
    # Hard caps. Both are graded. Do not raise them to make a test pass.
    max_tool_calls: int = 2
    max_corrections: int = 2
    request_timeout_s: float = 30.0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("MODEL", "gpt-5.4-mini-2026-03-17"),
    )


def has_key() -> bool:
    """True when a real key is configured. /health must work when this is False."""
    return bool(get_settings().openai_api_key)
