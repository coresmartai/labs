"""Bounded exponential backoff for tool-level retries.

Two flavors of retry live in this codebase:
  - Tool-level (this file): transient external failures. tenacity wraps the
    impl. 3 attempts, base 500ms, capped at 8s.
  - Model-level (in llm.py): pydantic.ValidationError when the model produced
    malformed args. Retry-with-correction pattern, cap 2.
"""
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
from app.config import get_settings


def with_tool_retry(fn):
    """Decorator: retry transient external failures with bounded backoff."""
    s = get_settings()
    return retry(
        stop=stop_after_attempt(s.tool_retry_max_attempts),
        wait=wait_exponential_jitter(
            initial=s.tool_retry_base_ms / 1000.0,
            max=s.tool_retry_max_ms / 1000.0,
        ),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True,
    )(fn)
