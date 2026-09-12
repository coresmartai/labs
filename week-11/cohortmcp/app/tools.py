"""Tool implementations for CohortMCP.

Two tools, both backed by real state on the file system (`app/data/*.json`),
not inline dicts - this is the "tool that depends on real state" half of
this week's build:

  - cohort_lookup(student_id)  -> profile
  - incident_history(query)    -> list[incident]

Pure functions, no MCP SDK import here at all - `app/mcp_server.py` is what
registers these onto a real FastMCP server. Keeping this module framework-
agnostic means every function here is directly callable and testable
without booting any server, and the OpenAI-facing eval in `app/llm.py`
reuses the exact same descriptions the MCP server exposes.

Descriptions come in a "good" and a "bad" variant, selected once at process
start by TOOL_DESCRIPTION_QUALITY. Flip it and restart to reproduce the
failure mode this week's eval harness (`app/eval.py`) is built to catch -
see `07_tool_description_quality_matrix.svg`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

# Fixtures live inside the package (app/data/) so an installed wheel finds them
# exactly like the source tree does - declared as package-data in pyproject.toml.
logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"


def _load_json(name: str) -> Any:
    with open(_DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


_COHORT: dict[str, dict[str, Any]] = _load_json("cohort.json")
_INCIDENTS: list[dict[str, Any]] = _load_json("incidents.json")


# ── Tool implementations ──────────────────────────────────────────────────────


def cohort_lookup(student_id: str) -> dict[str, Any]:
    profile = _COHORT.get(student_id)
    if profile is None:
        return {"found": False}
    return {"found": True, "profile": profile}


def incident_history(query: str) -> dict[str, Any]:
    q = query.lower()
    matches = [i for i in _INCIDENTS if q in i["summary"].lower()]
    return {"results": matches[:25], "total_available": len(matches)}


# ── Descriptions - good vs bad, toggled by TOOL_DESCRIPTION_QUALITY ──────────

_GOOD_DESCRIPTIONS = {
    "cohort_lookup": (
        "Look up a cohort member's profile by student_id. Use when an "
        "orchestrator needs to know which student's service handles a given "
        "specialty, or when surfacing the cohort directory to a user. Returns "
        "{'found': false} (not an error) when student_id has no match - do not "
        "retry on 'found: false', it means the id does not exist."
    ),
    "incident_history": (
        "Search recent incident history for entries whose summary contains the "
        "query text (case-insensitive substring match). Use when the user asks "
        "about past incidents, wants to see trends, or needs an incident id to "
        "reference. Returns up to 25 matches; an empty 'results' list (not an "
        "error) means nothing matched - try a broader query rather than retrying."
    ),
}

# Deliberately vague - verb missing, scope unstated, no guidance on the
# empty/not-found case. This is what "tool descriptions the model actually
# obeys" is warning against.
_BAD_DESCRIPTIONS = {
    "cohort_lookup": "Looks stuff up.",
    "incident_history": "Gets incidents.",
}


def tool_description(name: str, quality: str) -> str:
    """Pure lookup, no settings dependency - trivially testable, and it makes
    explicit that the choice is a function of `quality` alone. `app/mcp_server.py`
    resolves `quality` from settings exactly once, at import time, and passes it
    in here; flipping TOOL_DESCRIPTION_QUALITY takes effect on the next process
    start, same as every other pinned config value in this course.
    """
    table = _BAD_DESCRIPTIONS if quality == "bad" else _GOOD_DESCRIPTIONS
    return table[name]
