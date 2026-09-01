"""The model calls, the tool round-trip, and retry with correction.

The shape of a request, in order:

    1. Model call one, NOT streamed. The model reads the review and either
       emits a tool call or does not.
    2. Your code runs lookup_order and appends a tool-role message.
    3. Model call two, STREAMED. Fields arrive as they complete.
    4. Validate. On ValidationError, re-ask with the validator's own message
       attached. Cap at settings.max_corrections.

Two caps, two failures, two log tags. Do not collapse them into one counter:
a runaway tool loop and a model that cannot satisfy your schema need
different fixes, and one counter makes both undiagnosable.
"""
from typing import Any, AsyncIterator

from app.config import get_settings


class SchemaCorrectionExhausted(RuntimeError):
    """Raised when the model still fails validation after the last retry.

    When this happens, emit `validation_failed` and stop. Do NOT emit
    `routed`, and do not close the stream silently.
    """


async def resolve_lookup(review: str) -> tuple[list[dict[str, Any]], int]:
    """Model call one plus the tool round-trip.

    Returns (messages, attempts) where `messages` is the conversation so far,
    including any tool-role result, and `attempts` is how many times the model
    asked you to look something up. A review with no order reference must
    return attempts == 0.

    Raise ToolLoopExceeded if the model asks more than settings.max_tool_calls
    times.
    """
    raise NotImplementedError("TODO: model call one, then the tool round-trip")


async def stream_ticket(messages: list[dict[str, Any]]) -> AsyncIterator[tuple[str, Any]]:
    """Model call two. Yield (field_name, value) as each field completes.

    Yield control between frames (await asyncio.sleep(0)) so one slow consumer
    does not stall the loop.
    """
    raise NotImplementedError("TODO: streamed extraction call")
    yield  # pragma: no cover - makes this an async generator


async def validate_with_correction(raw: dict[str, Any],
                                   messages: list[dict[str, Any]]) -> tuple[Any, int]:
    """Validate `raw` against Ticket, re-asking on failure.

    Returns (ticket, corrections). Feed the ValidationError message back to
    the model on each retry: re-asking with an unchanged prompt gives the
    model no new information and fails identically.

    Raise SchemaCorrectionExhausted after settings.max_corrections.
    """
    settings = get_settings()  # noqa: F841 - you will need max_corrections
    raise NotImplementedError("TODO: retry with correction")
