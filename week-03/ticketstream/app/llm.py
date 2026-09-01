"""TicketStream LLM wrapper - async OpenAI SDK + structured-output via tool calling.

Two public functions:
  - _aextract_once: single non-streaming model call returning tool args.
  - validate_and_correct: retry-with-correction loop wrapping _aextract_once.
"""
import json
import logging
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import TicketSchema
from app.tools import EXTRACT_TICKET_TOOL

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are TicketStream's intake assistant.

Behavior contract:
- Always call extract_ticket. Do not produce prose.
- Use the priority enum literally - do not invent values like 'urgent' or 'asap'.
- Set customer_id to null if the user did not provide one.

Format directive:
- Respond by calling the extract_ticket tool with TicketSchema arguments.

If asked anything unrelated to support intake, still call extract_ticket
with intent='order' and priority='low' so downstream code can decide.
"""


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


async def _aextract_once(
    messages: list[dict[str, Any]],
    model_name: str,
) -> tuple[str | None, dict[str, Any] | None, str]:
    """Single model call. Returns (tool_name, tool_args, finish_reason)."""
    client = _client()
    response = await client.chat.completions.create(
        model=model_name,
        max_completion_tokens=1024,
        tools=[EXTRACT_TICKET_TOOL],
        messages=messages,
    )
    choice = response.choices[0]
    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        tc = choice.message.tool_calls[0]
        return tc.function.name, json.loads(tc.function.arguments), choice.finish_reason
    return None, None, choice.finish_reason


async def validate_and_correct(
    user_message: str,
    model_name: str | None = None,
) -> tuple[TicketSchema | None, dict[str, int]]:
    """Run extract_ticket with retry-with-correction.

    Returns (ticket, telemetry).
      - ticket = None means the model never produced a valid Ticket.
      - telemetry includes correction_count, attempts.
    """
    s = get_settings()
    model = model_name or s.model_name
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    correction_count = 0

    for attempt in range(s.schema_correction_max_attempts + 1):
        name, args, finish_reason = await _aextract_once(messages, model)

        if name is None:
            logger.warning(
                "model emitted text instead of tool_calls",
                extra={"event": "unexpected_text_response"},
            )
            return None, {"correction_count": correction_count,
                          "attempts": attempt + 1}

        try:
            ticket = TicketSchema.model_validate(args)
            return ticket, {"correction_count": correction_count,
                            "attempts": attempt + 1}
        except ValidationError as e:
            if attempt >= s.schema_correction_max_attempts:
                logger.error("schema correction exhausted",
                             extra={"event": "schema_correction_exhausted",
                                    "errors": e.errors()})
                return None, {"correction_count": correction_count,
                              "attempts": attempt + 1}

            correction_count += 1
            logger.info("retry-with-correction firing",
                        extra={"event": "tool_arg_validation_failed",
                               "errors": e.errors()})
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "extract_ticket",
                        "arguments": json.dumps(args),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": "t1",
                "content": (
                    f"Your previous arguments failed validation:\n"
                    f"{json.dumps(e.errors(), default=str)}\n"
                    "Please correct and retry."
                ),
            })

    return None, {"correction_count": correction_count,
                  "attempts": s.schema_correction_max_attempts + 1}
