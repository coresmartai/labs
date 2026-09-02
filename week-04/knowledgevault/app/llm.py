"""
Thin SDK wrapper for the OpenAI vision LLM (gpt-5.4-mini-2026-03-17).

Why wrap the SDK?
  - The rest of the pipeline calls *our* function, not the SDK. Provider
    swaps (or test mocks) survive.
  - Centralizes model pinning, retries, structured-output enforcement.
  - Keeps the function-calling round-trip in one place.

Public surface:
  - describe_figure(image_bytes: bytes) -> FigureDescription
"""

from __future__ import annotations
import base64
import json
import logging
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.schemas import FigureDescription
from app.tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_VISION = """You are an engineering documentation assistant.
For each figure you see, use the record_figure_description function to record a
structured description. Be concrete. Prefer 'low' confidence over guessing
unlabeled chart numbers. Describe what is shown, not what should be shown.
"""


def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def describe_figure(image_bytes: bytes, media_type: str = "image/png") -> FigureDescription:
    """
    Send a single figure image to gpt-5.4-mini-2026-03-17 and return a validated
    FigureDescription. Forces the function-calling pattern so the output shape
    is pinned regardless of what the model would otherwise emit.

    Raises ValueError if the model declines to call the function (rare; means
    the image was unreadable or the model refused).
    """
    settings = get_settings()
    client = _client()

    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_url = f"data:{media_type};base64,{b64}"

    resp = client.chat.completions.create(
        model=settings.openai_vision_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_VISION},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Describe this engineering figure."},
                ],
            },
        ],
        tools=TOOL_SCHEMAS,
        tool_choice={"type": "function", "function": {"name": "record_figure_description"}},
    )

    message = resp.choices[0].message
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.function.name == "record_figure_description":
                args: dict[str, Any] = json.loads(tool_call.function.arguments)
                execute_tool(tool_call.function.name, args)
                try:
                    return FigureDescription.model_validate(args)
                except Exception as e:  # noqa: BLE001
                    logger.error("FigureDescription validation failed: %s", e)
                    raise ValueError(
                        f"Vision LLM returned malformed args: {e}\nargs: {args}"
                    )

    raise ValueError("Vision LLM did not call record_figure_description")
