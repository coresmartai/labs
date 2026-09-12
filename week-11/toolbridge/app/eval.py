"""The tool-pick eval. One number, and it moves when you change a description.

    python -m app.eval                 # scores the current description set
    TOOL_DESCRIPTION_QUALITY=bad python -m app.eval

Run it twice and write down both numbers. That is the whole exercise: it turns
"description quality is the biggest lever" from something you were told into
something you measured on your own machine.

Three things this scorer does that a naive one does not, all of them from W11-R04:

  * It scores SELECTION only. A wrong argument is a schema problem and a wrong
    tool is a description problem, and one number hides which.
  * The golden set contains rows whose correct answer is to call NOTHING. A set
    without them optimises straight into a server that fires on everything.
  * It reports per-row, so a failure is a line you can read rather than a
    percentage you can only stare at.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.descriptions import describe

_DATA = Path(__file__).parent / "data" / "golden_tool_picks.json"
TOOL_NAMES = ["peer_card", "delegate_triage", "resume_task"]


def load_rows() -> list[dict[str, Any]]:
    return json.loads(_DATA.read_text())["rows"]


def _tool_specs(quality: str) -> list[dict[str, Any]]:
    """The tool list exactly as a host would serialise it into model context."""
    return [{"name": n, "description": describe(n, quality)} for n in TOOL_NAMES]


async def pick_tool(request: str, quality: str) -> str:
    """Ask a model which tool to call, or none. Returns a tool name or 'none'.

    Deliberately NOT wired through the MCP server: we are measuring the
    descriptions, so the descriptions are the only thing that varies.
    """
    from openai import AsyncOpenAI  # imported here so tests never need the package

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    tools = [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"],
                      "parameters": {"type": "object", "properties": {}}}}
        for t in _tool_specs(quality)
    ]
    response = await client.chat.completions.create(
        model=settings.picker_model,
        messages=[
            {"role": "system",
             "content": "You route a user request to at most one tool. If no tool fits, answer without calling one."},
            {"role": "user", "content": request},
        ],
        tools=tools,
        tool_choice="auto",  # NOT "required": rows whose answer is none must be reachable
    )
    calls = response.choices[0].message.tool_calls or []
    return calls[0].function.name if calls else "none"


async def run_eval(quality: str | None = None) -> dict[str, Any]:
    quality = quality or get_settings().tool_description_quality
    rows = load_rows()
    results = []
    for row in rows:
        try:
            got = await pick_tool(row["request"], quality)
        except Exception as exc:  # a model failure is an eval failure, not a crash
            got = f"error:{type(exc).__name__}"
        results.append({"request": row["request"], "expected": row["expected"],
                        "got": got, "ok": got == row["expected"], "why": row["why"]})
    correct = sum(1 for r in results if r["ok"])
    return {"quality": quality, "total": len(results), "correct": correct,
            "accuracy": round(correct / len(results), 3), "rows": results}


def main() -> None:
    quality = os.environ.get("TOOL_DESCRIPTION_QUALITY") or get_settings().tool_description_quality
    summary = asyncio.run(run_eval(quality))
    print(f"\ntool-pick accuracy, descriptions={summary['quality']}: "
          f"{summary['correct']}/{summary['total']} = {summary['accuracy']:.1%}\n")
    for r in summary["rows"]:
        mark = "  ok " if r["ok"] else "FAIL "
        print(f"{mark} {r['expected']:<16} got {r['got']:<16} {r['request'][:58]}")
        if not r["ok"]:
            print(f"       why this row exists: {r['why']}")


if __name__ == "__main__":
    main()
