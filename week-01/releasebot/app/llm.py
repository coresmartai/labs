"""
LLM client wrapper.

Why wrap the SDK?
  - The rest of the code calls *our* function, not the SDK's. That makes
    provider swaps (or mocks in tests) survivable.
  - Centralizes model pinning, retry policy, and logging in one place.
  - Lets us add tool-call orchestration without leaking it everywhere.

Public surface:
  - stream_text(prompt: str)               -> async generator of text deltas
  - summarize_with_tools(notes, recipient) -> dict with summary + tool_calls
"""

from __future__ import annotations
import json
import logging
from typing import AsyncIterator

from openai import OpenAI, AsyncOpenAI

from app.config import get_settings
from app.schemas import ReleaseSummary
from app.tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Client construction (sync + async)
# ------------------------------------------------------------------

def _async_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


def _sync_client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


# ------------------------------------------------------------------
# Streaming - used by /summarize-stream
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are ReleaseBot, an assistant that summarizes release notes for "
    "engineering teams. Be concise. Lead with the impact. Avoid marketing fluff."
)


async def stream_text(prompt: str) -> AsyncIterator[str]:
    """Yield text deltas from the model as they arrive (SSE source)."""
    settings = get_settings()
    client   = _async_client()

    stream = await client.chat.completions.create(
        model=settings.openai_model,
        max_completion_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def summarize_with_tools(release_notes: str, recipient: str) -> dict:
    """
    One API call:
      1. Model receives release notes + send_email schema.
      2. tool_choice forces it to fill headline/bullets/risk_level as args.
      3. We execute send_email, returning the email result.
      4. We build ReleaseSummary directly from the tool args - no parsing.
    """
    settings = get_settings()
    client   = _sync_client()

    logger.info("[summarize] model=%s  recipient=%s", settings.openai_model, recipient)

    resp = client.chat.completions.create(
        model=settings.openai_model,
        max_completion_tokens=512,
        tools=TOOL_SCHEMAS,
        tool_choice={"type": "function", "function": {"name": "send_email"}},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Summarize these release notes and send the summary "
                    f"to {recipient} using the send_email tool.\n\n"
                    f"Release notes:\n{release_notes}"
                ),
            },
        ],
    )

    finish = resp.choices[0].finish_reason
    logger.info("[summarize] finish_reason=%s", finish)

    tool_calls: list[dict] = []
    summary: ReleaseSummary | None = None

    if finish == "tool_calls" and resp.choices[0].message.tool_calls:
        for tc in resp.choices[0].message.tool_calls:
            args = json.loads(tc.function.arguments)
            logger.info(
                "[summarize] tool=%s  headline=%.60r  risk=%s  bullets=%d",
                tc.function.name, args.get("headline"), args.get("risk_level"),
                len(args.get("bullets", [])),
            )

            result = execute_tool(tc.function.name, args)
            logger.info("[summarize] tool result=%s", result)
            tool_calls.append({"name": tc.function.name, "input": args, "result": result})

            # Build ReleaseSummary directly from the tool args - no text parsing.
            if summary is None and tc.function.name == "send_email":
                summary = ReleaseSummary(
                    headline   = args["headline"],
                    bullets    = args["bullets"],
                    risk_level = args["risk_level"],
                )
    else:
        logger.warning(
            "[summarize] tool not triggered  finish_reason=%s  content=%.120r",
            finish, resp.choices[0].message.content,
        )

    if summary is None:
        raise ValueError(
            f"Model did not call send_email (finish_reason={finish!r}). "
            "Cannot produce a structured summary."
        )

    return {"summary": summary.model_dump(), "tool_calls": tool_calls}
