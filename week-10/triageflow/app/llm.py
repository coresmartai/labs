"""Thin wrapper around the OpenAI SDK.

Route every model call through this module so the rest of the codebase
remains provider-agnostic. Pinned model versions live in config.Settings.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, APIConnectionError, InternalServerError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# Transient failures worth retrying: network drops, rate limits, 5xx.
_RETRYABLE = (APIConnectionError, RateLimitError, InternalServerError)


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=8.0),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 1024,
) -> LLMResult:
    """Single synchronous call, retried by tenacity on transient errors
    (3 attempts, exponential backoff with jitter 0.5s -> 8s, reraise).

    Returns a structured result so callers don't depend on the SDK shape.
    """
    # OpenAI chat completions: system prompt is just the first message
    full_messages = [{"role": "system", "content": system}] + messages
    start = time.monotonic()
    resp = _client().chat.completions.create(
        model=model,
        messages=full_messages,
        max_completion_tokens=max_tokens,
    )
    elapsed = int((time.monotonic() - start) * 1000)
    text = resp.choices[0].message.content or ""
    logger.info(
        "llm.call model=%s latency_ms=%d input_tokens=%d output_tokens=%d",
        model, elapsed, resp.usage.prompt_tokens, resp.usage.completion_tokens,
    )
    return LLMResult(
        text=text,
        model=model,
        latency_ms=elapsed,
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )
