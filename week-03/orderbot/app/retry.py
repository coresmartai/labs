"""Bounded exponential backoff for tool-level retries.

Two flavors of retry live in this codebase:
  - Tool-level (this file): transient external failures. tenacity wraps the
    impl. 3 attempts, base 500ms, capped at 8s.
  - Model-level (in llm.py): pydantic.ValidationError when the model produced
    malformed args. Retry-with-correction pattern, cap 2.
"""
import logging

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


def _log_transient_failure(retry_state) -> None:
    """tenacity before_sleep hook - emit the tool_impl_transient_failed tag.

    Every backoff is a structured event, not a silent pause. This is the tag
    you alert on: a rising tool_impl_transient_failed rate means the service
    underneath your tool is degrading, and the retries are hiding it from
    your users right up until they stop working.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "transient tool failure on attempt %d; backing off %.2fs",
        retry_state.attempt_number,
        getattr(retry_state.next_action, "sleep", 0.0),
        extra={
            "event": "tool_impl_transient_failed",
            "attempt": retry_state.attempt_number,
            "error": type(exc).__name__ if exc else None,
        },
    )


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
        before_sleep=_log_transient_failure,
        reraise=True,
    )(fn)
