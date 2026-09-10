"""Thin OpenAI SDK wrapper.

Route handlers never call the SDK directly - they go through
`call_with_tools`. This is the seam where the provider can be
swapped with a single config change (vLLM, Azure OpenAI, etc.)
as long as it exposes the OpenAI-compatible chat completions API.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# Week-3 retry canon: 3 attempts, exponential backoff with jitter,
# network-ish errors only, re-raise the final failure. Application-layer
# errors (4xx/5xx bodies, schema problems) are NOT retried here.
#
# This is the only retry layer between the loop and the MODEL PROVIDER. The
# SDK's own retries are disabled where the client is constructed - see the
# comment there. tools.py carries the same policy against a different
# dependency, the tools' own HTTP calls, which is one layer each rather than
# two stacked. That distinction is the whole of the counting rule: count
# layers per dependency, because only stacked layers multiply. W09-R01 calls
# the alternative retry amplification - three layers of three attempts is
# twenty-seven calls to a dependency that is already unhealthy.
NETWORK_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    httpx.TimeoutException,
    httpx.NetworkError,
)


@dataclass
class ModelResponse:
    """Provider-agnostic shape for a single model turn.

    The agent loop in agent.py reads only these fields, so swapping
    providers is a question of writing a new adapter that produces
    this shape.
    """
    content_blocks: list[dict]   # Normalised assistant content (text + tool_call).
    stop_reason: str             # "end_turn" or "tool_calls"
    input_tokens: int
    output_tokens: int
    # The slice of input_tokens the provider served from its prompt cache.
    # Billed at 10% of the input rate - see config.MODEL_PRICES.
    cached_input_tokens: int = 0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=8.0),
    retry=retry_if_exception_type(NETWORK_ERRORS),
    reraise=True,
)
def _create_completion(client: OpenAI, **kwargs):
    """One SDK call, wrapped with the Week-3 retry canon."""
    return client.chat.completions.create(**kwargs)


def call_with_tools(
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
) -> ModelResponse:
    """Run one turn of the model with the tool catalog.

    Prepends the system prompt as a system-role message, calls the
    OpenAI chat completions API, and normalises the response into a
    ModelResponse so the agent loop stays provider-agnostic.
    """
    settings = get_settings()
    # max_retries=0 turns the SDK's own retry layer OFF, on purpose.
    # The SDK defaults to 2 retries, which is 3 attempts. Leave that on and
    # the tenacity wrapper below multiplies with it: 3 x 3 = 9 model calls
    # from one iteration, aimed at a provider that is already unhealthy.
    # That is retry amplification, and this is the one line that removes it.
    # The policy lives in _create_completion, where you can read it.
    client = OpenAI(api_key=settings.openai_api_key, max_retries=0)

    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = _create_completion(
        client,
        model=settings.openai_model,
        max_completion_tokens=settings.max_completion_tokens,
        messages=full_messages,
        tools=tools,
        timeout=settings.request_timeout_seconds,
    )

    choice = response.choices[0]
    msg = choice.message

    # Normalise to the provider-agnostic content_blocks shape.
    blocks: list[dict] = []
    if msg.content:
        blocks.append({"type": "text", "text": msg.content})
    if msg.tool_calls:
        for tc in msg.tool_calls:
            blocks.append({
                "type": "tool_call",
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments),
            })

    stop_reason = "end_turn" if choice.finish_reason == "stop" else "tool_calls"

    # Cached-prompt tokens, when the provider reports them. Not every
    # OpenAI-compatible endpoint does, so read it defensively.
    details = getattr(response.usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0

    return ModelResponse(
        content_blocks=blocks,
        stop_reason=stop_reason,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cached_input_tokens=cached,
    )
