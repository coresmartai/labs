"""
Tool schemas + dispatcher for the vision LLM round-trip.

The vision describer asks the vision LLM (gpt-5.4-mini) to call a single function,
`record_figure_description`, whose `parameters` schema matches `FigureDescription`.
Pinning the output shape through `tool_choice` is what saves us from failure #1 -
naive prompts return prose, forced function calls return validated dicts.

Two questions to ask of every tool:
  1. What if the model calls it twice?       (idempotency)
  2. What's the blast radius on bad input?   (safety)

For this tool the answer is trivial: it just records a description. No side
effects. The blast radius is "one bad row in the chunker output."
"""

from __future__ import annotations
from typing import Any
import logging

logger = logging.getLogger(__name__)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "record_figure_description",
            "description": (
                "Record a structured description of an engineering figure. "
                "Use this for every figure, chart, screenshot, or diagram the user shows you. "
                "Be precise. Prefer 'low' confidence over guessing chart numbers."
            ),
            # strict mode: the API constrains generation to this schema. Strict mode
            # requires additionalProperties: false and every property listed in
            # `required`. Unsupported keywords (maxLength/maxItems) are intentionally
            # absent - Pydantic enforces length/item limits on our side of the wire.
            "strict": True,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {
                        "type": "string",
                        "description": (
                            "Short tag, e.g. 'architecture diagram', 'sequence diagram', "
                            "'chart', 'screenshot', 'table image'."
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": "2–4 sentence description of the figure's content.",
                    },
                    "key_elements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels, boxes, arrows, or salient elements named in the figure.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "How confident you are in the description. 'low' = legible issues "
                            "or made-up axes; downstream may drop low-confidence descriptions."
                        ),
                    },
                },
                "required": ["type", "summary", "key_elements", "confidence"],
            },
        },
    }
]


def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatcher. The vision describer is single-tool; if we add more tools in
    future weeks, this is the one place that grows.
    """
    if name == "record_figure_description":
        # The "execution" here is just echoing back - the vision describer
        # collects the structured input and passes it to the chunker.
        logger.info("Tool record_figure_description called with type=%s, confidence=%s",
                    args.get("type"), args.get("confidence"))
        return {"success": True, "recorded": args}
    return {"success": False, "error": "unknown_tool", "tool": name}
