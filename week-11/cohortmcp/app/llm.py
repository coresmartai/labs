"""Thin OpenAI wrapper - the eval harness's only LLM touchpoint.

Given a natural-language request, ask the pinned model to pick one of this
week's two MCP tools via function calling, so `app/eval.py` can measure
whether good vs bad tool descriptions change pick accuracy.

The function-calling schema is built from `app/mcp_server.py`'s live tool
registration rather than a second, hand-maintained copy - the MCP tools and
the eval always see the exact same names, schemas, and descriptions.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.mcp_server import list_tools
from app.schemas import ToolPickResponse

logger = logging.getLogger(__name__)


async def _openai_tool_schemas() -> list[dict[str, Any]]:
    tools = await list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in tools
    ]


async def pick_tool(request: str) -> ToolPickResponse:
    """Ask the model which tool it would call for `request`. Never raises past
    a malformed tool-call payload - a missing tool call just means `tool: None`.
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set - required for /eval/pick (tool-pick eval). "
            "The MCP tool server itself runs without it."
        )
    # base_url=None -> the SDK's default endpoint. Set OPENAI_BASE_URL and the same
    # code talks to any OpenAI-compatible endpoint (gateway, proxy, self-hosted).
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    start = time.perf_counter()

    # tool_choice="required" is what forces determinism of *shape* (exactly one tool
    # call, arguments validated against the tool's own JSON Schema); temperature=0
    # pins the sampling. The eval scores the pick, not the prose.
    completion = client.chat.completions.create(
        model=settings.openai_model,
        max_completion_tokens=256,
        temperature=0,
        tools=await _openai_tool_schemas(),
        tool_choice="required",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an MCP host deciding which tool to call for the user's "
                    "request. Call exactly one tool with the correct arguments."
                ),
            },
            {"role": "user", "content": request},
        ],
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    message = completion.choices[0].message
    calls = message.tool_calls or []
    if not calls:
        logger.warning("llm.no_tool_call request=%r", request)
        return ToolPickResponse(tool=None, arguments={}, latency_ms=latency_ms)

    call = calls[0]
    try:
        arguments = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        arguments = {}

    logger.info("llm.tool_picked tool=%s latency_ms=%s", call.function.name, latency_ms)
    return ToolPickResponse(tool=call.function.name, arguments=arguments, latency_ms=latency_ms)
