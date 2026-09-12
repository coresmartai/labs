"""Tool implementations for the cohort MCP server.

Four tools, all backed by real state on the file system (`app/data/*.json`) or by
this instance's configured identity, not inline dicts:

  - cohort_lookup(student_id)  -> profile
  - incident_history(query)    -> list[incident]
  - whoami()                   -> who runs this agent
  - ping(message)              -> round-trip check

The last two exist because someone else is on the other end of the connection.
`cohort_lookup` answers about you as well as about the fixtures, so a classmate
who looks up your id gets you rather than `{'found': false}`.

Pure functions, no MCP SDK import here at all - `app/mcp_server.py` is what
registers these onto a real FastMCP server. Keeping this module framework-
agnostic means every function here is directly callable and testable without
booting any server.

Descriptions come in a "good" and a "bad" variant, selected once at process
start by TOOL_DESCRIPTION_QUALITY. Flipping it changes what a *classmate's*
model reads when it connects to your server - the description is the prompt,
and here you get to watch someone else's model obey it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.identity import student_identity

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
    # Answer about yourself first. The fixtures list Alice, Bob and Cleo; they do
    # not list *you*, and a classmate who looks up your real id should not be told
    # you do not exist.
    if student_id == get_settings().student_id:
        me = student_identity()
        return {"found": True, "profile": {
            "name": me["student_name"], "specialty": me["specialty"], "host": me["base_url"]}}
    profile = _COHORT.get(student_id)
    if profile is None:
        return {"found": False}
    return {"found": True, "profile": profile}


def incident_history(query: str) -> dict[str, Any]:
    q = query.lower()
    matches = [i for i in _INCIDENTS if q in i["summary"].lower()]
    return {"results": matches[:25], "total_available": len(matches)}


def all_incidents() -> list[dict[str, Any]]:
    """The whole incident list, unfiltered - what the `agentmesh://incidents`
    resource serves. A resource is not a tool, so this is not registered as one;
    it exists so `app/mcp_server.py` never has to reach for a private name."""
    return _INCIDENTS


# ──────────────────────────────────────────────────────────────────────────────
# Identity and connectivity. Both of these exist because someone else is on the
# other end of the connection; on one laptop, alone, neither would mean anything.


def whoami() -> dict[str, Any]:
    return student_identity()


def ping(message: str = "ping") -> dict[str, Any]:
    me = student_identity()
    return {"echo": message, "from": me["student_id"], "student_name": me["student_name"]}


# ── Descriptions - good vs bad, toggled by TOOL_DESCRIPTION_QUALITY ──────────

_GOOD_DESCRIPTIONS = {
    "cohort_lookup": (
        "Look up a cohort member's AgentMesh profile by student_id. Use when an "
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
    "whoami": (
        "Identify the student who owns this agent. Use this first when you connect "
        "to an unfamiliar cohort server - it answers 'whose agent am I talking to?' "
        "without needing any other tool. Returns student_id, student_name, "
        "specialty, the agent's base URL, and which protocols this instance speaks. "
        "Takes no arguments and never fails."
    ),
    "ping": (
        "Echo a message back with this agent's identity attached. Use as a "
        "connectivity and latency check against a classmate's server before "
        "submitting real work - it proves the transport, the auth header and the "
        "tool dispatch all work, without touching any business logic."
    ),
}

# Deliberately vague - verb missing, scope unstated, no guidance on the
# empty/not-found case. This is what "tool descriptions the model actually
# obeys" is warning against.
_BAD_DESCRIPTIONS = {
    "cohort_lookup": "Looks stuff up.",
    "incident_history": "Gets incidents.",
    "whoami": "Info about this.",
    "ping": "Pings.",
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
