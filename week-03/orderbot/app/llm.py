"""Thin OpenAI SDK wrapper.

Route handlers MUST NOT touch the SDK directly. They call call_with_tools().
"""
import json
import logging
import time
from typing import Any
from openai import OpenAI

from app.config import get_settings
from app.tools import tool_definitions, execute_tool
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def _client() -> OpenAI:
    return OpenAI(api_key=get_settings().openai_api_key)


SYSTEM_PROMPT = """You are an order-lookup assistant.

Behavior contract:
- Always call lookup_order_status when the user asks about an order.
- Do not invent order IDs, ETAs, or shipping states.
- If you cannot determine an order ID from the user's message, ask for it.

Format directive:
- Use the lookup_order_status tool. Do not respond in prose without first
  calling the tool when the question is about an order.
"""


def call_with_tools(
    user_message: str,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Run the four-message round-trip with bounded retries.

    Returns a dict with answer text, telemetry counts, and latency_ms.
    Raises LoopExceededError if the model exceeds the iteration cap.
    """
    s = get_settings()
    client = _client()
    model = model_name or s.model_name
    t0 = time.monotonic()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    tool_call_count = 0
    correction_count = 0

    for _iteration in range(s.tool_loop_max_iterations):
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=1024,
            tools=tool_definitions(),
            messages=messages,
        )
        choice = response.choices[0]

        # Did the model use a tool?
        if choice.finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls or []
            if not tool_calls:
                logger.warning("finish_reason=tool_calls but no tool_calls",
                               extra={"event": "unexpected_text_response"})
                break

            # Append the assistant's message (with tool_calls) to history.
            messages.append({
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tc in tool_calls:
                tool_call_count += 1
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                try:
                    result = execute_tool(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result),
                    })
                except ValidationError as e:
                    if correction_count >= s.schema_correction_max_attempts:
                        logger.error(
                            "retry-with-correction exhausted after %d attempts",
                            correction_count,
                            extra={"event": "schema_correction_exhausted",
                                   "tool": name},
                        )
                        raise SchemaCorrectionExhausted(
                            "Model produced malformed args twice; check prompt"
                        )
                    correction_count += 1
                    logger.info("retry-with-correction firing",
                                extra={"event": "tool_arg_validation_failed",
                                       "errors": e.errors()})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            f"Your previous arguments failed validation: "
                            f"{e.errors()}. Please correct and retry."
                        ),
                    })
            continue

        # Model is done - extract text.
        text = choice.message.content or ""
        return {
            "answer": text,
            "tool_call_count": tool_call_count,
            "retry_count": correction_count,
            "latency_ms": round((time.monotonic() - t0) * 1000, 1),
        }

    # The model kept asking for tools past the cap. This is the runaway-agent
    # guard: log the tag first, then raise, so the event is on the wire even
    # if the caller swallows the exception.
    logger.error(
        "tool loop hit the iteration cap of %d",
        s.tool_loop_max_iterations,
        extra={"event": "loop_exceeded",
               "tool_call_count": tool_call_count},
    )
    raise LoopExceededError("Exceeded tool-loop iteration cap")


class LoopExceededError(Exception):
    """Raised when the model tries to call more tools than tool_loop_max_iterations."""


class SchemaCorrectionExhausted(Exception):
    """Raised when retry-with-correction hit its cap; surface to user."""
