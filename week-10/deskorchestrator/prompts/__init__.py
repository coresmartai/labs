"""Prompts are content, and content gets versioned. They live on disk."""
from __future__ import annotations

from pathlib import Path

from app.memory import get_prefs

_HERE = Path(__file__).parent


def _load(name: str) -> str:
    return (_HERE / f"{name}.txt").read_text(encoding="utf-8").strip()


TRIAGE_BASE = _load("triage")
ACCESS_BASE = _load("access")
SOFTWARE_BASE = _load("software")
HARDWARE_BASE = _load("hardware")

SPECIALIST_BASE = {"access": ACCESS_BASE, "software": SOFTWARE_BASE, "hardware": HARDWARE_BASE}


def compose_system_prompt(user_id: str, base_prompt: str) -> str:
    """Fold this user's procedural rules into any agent's system prompt.

    Easy to write and easy to leave dead: if nothing ever writes a preference the
    dict stays empty, no rule is ever injected, and the code still looks like it
    works. `memory.seed_demo_prefs` is the writer that keeps this layer alive.
    """
    prefs = get_prefs(user_id)
    if not prefs:
        return base_prompt
    rules = []
    if prefs.get("preferred_response_length") == "concise":
        rules.append("- Keep the answer to three sentences or fewer.")
    if prefs.get("preferred_response_length") == "detailed":
        rules.append("- Give the full reasoning, not a summary.")
    if prefs.get("always_cc_manager_on_access"):
        rules.append("- For any access request, state explicitly that the manager will be copied.")
    if not rules:
        return base_prompt
    return base_prompt + "\n\nUser preferences (procedural memory):\n" + "\n".join(rules)
