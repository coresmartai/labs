"""Prompt assembly - loads agent system prompts and folds in procedural memory.

Prompts live in their own .txt files so they can be Git-versioned cleanly

"""
from pathlib import Path
from typing import Any

from app.memory import get_prefs

_HERE = Path(__file__).parent


def _load(name: str) -> str:
    return (_HERE / f"{name}.txt").read_text(encoding="utf-8").strip()


TRIAGE_BASE = _load("triage")
KNOWLEDGE_BASE = _load("knowledge")
ACTION_BASE = _load("action")


def compose_system_prompt(user_id: str, base_prompt: str) -> str:
    """Prepend the user's procedural memory rules to the agent's base prompt."""
    prefs = get_prefs(user_id)
    if not prefs:
        return base_prompt

    rules: list[str] = []
    if prefs.get("always_escalate_billing"):
        rules.append("- Always escalate billing-related incidents to a human reviewer.")
    if prefs.get("preferred_response_length") == "concise":
        rules.append("- Keep responses concise; bullet points only when essential.")
    blocked: list[str] | None = prefs.get("blocked_tools")
    if blocked:
        rules.append(f"- The following tools are forbidden for this user: {', '.join(blocked)}.")

    if not rules:
        return base_prompt
    return base_prompt + "\n\nUser preferences (procedural memory):\n" + "\n".join(rules)
