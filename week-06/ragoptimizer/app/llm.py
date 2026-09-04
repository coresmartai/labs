"""Thin SDK wrapper - every model call in the service goes through here.

Pins the model version. Wraps both sync and async paths. Business code
never imports the OpenAI SDK directly - it imports from this module.
"""
from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI, OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

# Week 5's refusal text, carried over verbatim. The W5 baseline in /eval/compare
# refuses with this exact string, so the 'before' column says what W5 said.
REFUSAL_STRING = "I don't have that information in the provided sources."


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


def _async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


def generate_text(
    prompt: str,
    *,
    max_tokens: int = 512,
    system: str | None = None,
    model: str | None = None,
) -> str:
    """Sync, non-streaming text generation. Used by HyDE."""
    settings = get_settings()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = _client().chat.completions.create(
        model=model or settings.model_id,
        max_completion_tokens=max_tokens,
        messages=messages,
    )
    return resp.choices[0].message.content or ""


def generate_json(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 1024,
    model: str | None = None,
) -> dict[str, Any]:
    """Sync JSON-mode generation - response is always parseable JSON.

    OpenAI requires the word 'json' (case-insensitive) to appear somewhere in
    the messages when response_format=json_object is set; our citation prompt
    satisfies this. Uses max_completion_tokens (new API name, same as max_tokens).
    """
    import json as _json
    settings = get_settings()
    resp = _client().chat.completions.create(
        model=model or settings.model_id,
        max_completion_tokens=max_tokens,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    logger.debug("generate_json raw response: %s", content[:200])
    return _json.loads(content)
