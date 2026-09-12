"""Golden-set eval - measures whether the model picks the right tool for a
request, not just any tool. Flip TOOL_DESCRIPTION_QUALITY in
`.env` to see the accuracy swing this week's diagrams describe: good
descriptions should score noticeably higher than bad ones on the same
8-example golden set.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.llm import pick_tool
from app.schemas import EvalResult, EvalSummary

logger = logging.getLogger(__name__)

_GOLDEN_PATH = Path(__file__).parent / "data" / "golden_tool_picks.json"


def load_golden() -> list[dict[str, Any]]:
    with open(_GOLDEN_PATH, encoding="utf-8") as f:
        return json.load(f)


async def run_eval() -> EvalSummary:
    """Run every golden example through `pick_tool`, score, and aggregate."""
    settings = get_settings()
    golden = load_golden()
    results: list[EvalResult] = []
    correct = 0
    for example in golden:
        picked = await pick_tool(example["request"])
        is_correct = picked.tool == example["expected_tool"]
        correct += int(is_correct)
        results.append(
            EvalResult(
                id=example["id"],
                request=example["request"],
                expected_tool=example["expected_tool"],
                picked_tool=picked.tool,
                correct=is_correct,
            )
        )
    total = len(golden)
    logger.info("eval.run quality=%s correct=%d/%d", settings.tool_description_quality, correct, total)
    return EvalSummary(
        quality=settings.tool_description_quality,
        model=settings.openai_model,
        total=total,
        correct=correct,
        accuracy=round(correct / total, 3) if total else 0.0,
        results=results,
    )
