"""Thin OpenAI SDK wrapper - the provider seam.

This is the only file in AgentMesh that imports the OpenAI SDK. Handlers never
call a provider directly; they go through here. That keeps the dated model pins
and the routing rule in one place.

Honesty note, and it stays true for the whole video: **the triage steps run
deterministic rules. Nothing in this repo calls a model at runtime.** Each step
below - `classify_incident`, `extract_action`, `synthesize_answer` - carries the
exact `complete()` call it would make, commented out right above the rule that
runs; uncomment it to dial the model (the functions are already `async`, so
nothing else changes). `model_for()` is what /health reports, so the pins on the
chip are the pins this file would dial.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from openai import APIError, OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)

Role = Literal["triage", "knowledge"]


class LLMClient:
    """Thin OpenAI client. Sync SDK called from async code via to_thread."""

    def __init__(self) -> None:
        settings = get_settings()
        self._api_key = settings.openai_api_key
        self._base_url = settings.openai_base_url          # None -> the SDK's default endpoint
        self._triage_model = settings.triage_model
        self._knowledge_model = settings.knowledge_model
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """Built on first use, not at import. /health only needs `model_for()`, and a
        liveness probe should never fail because a provider client could not be
        constructed."""
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def model_for(self, role: Role) -> str:
        """Which pinned model a given role dials. Public and testable - /health
        builds its `models` block from this, so the chip cannot drift from the seam."""
        return self._triage_model if role == "triage" else self._knowledge_model

    async def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        role: Role = "knowledge",
    ) -> str:
        """Plain completion.

        role="triage"    -> the cheap nano pin
        role="knowledge" -> the structured-output mini pin (default)

        No `temperature` is sent here, and nothing is being traded away: the triage
        path is decided by `triage_core.py`'s deterministic rules, not by a model, so
        there is no sampling to tighten in the first place.
        """
        model = self.model_for(role)
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=[{"role": "system", "content": system}] + messages,
                max_completion_tokens=1024,
            )
            return response.choices[0].message.content or ""
        except APIError as e:
            logger.error("llm.api_error error=%s model=%s role=%s", str(e), model, role)
            raise

    # ---- Triage agent steps (called from app/triage_core.py) ----------------
    # Each step runs a DETERMINISTIC rule, so the whole session is offline,
    # reproducible, and needs no real key. Directly above each rule is the real
    # model call it replaces: uncomment the `await self.complete(...)` block and
    # drop the rule below it to make this build call OpenAI. The functions are
    # already `async`, so flipping the comment is the only edit needed.

    async def classify_incident(self, description: str, severity: str) -> str:
        """Route an incident to 'knowledge', 'action', or 'escalate'."""
        # --- LLM version (uncomment to go live) ------------------------------
        # verdict = await self.complete(
        #     role="triage",
        #     system="You are an incident-triage router. Reply with exactly one word: "
        #            "knowledge, action, or escalate.",
        #     messages=[{"role": "user", "content": f"severity={severity}\n{description}"}],
        # )
        # return verdict.strip().lower()
        # --- deterministic rule (what runs) ----------------------------------
        desc = description.lower()
        if severity in ("critical",):
            return "escalate"
        if any(kw in desc for kw in ["how", "where", "what is", "runbook"]):
            return "knowledge"
        if any(kw in desc for kw in ["restart", "rotate", "scale", "rollback", "deploy"]):
            return "action"
        return "knowledge"

    async def extract_action(self, description: str) -> dict[str, str]:
        """Pull the remediation verb + target out of an action request."""
        # --- LLM version (uncomment to go live) ------------------------------
        # import json
        # raw = await self.complete(
        #     role="triage",
        #     system='Extract the remediation and target as JSON: '
        #            '{"remediation": "restart|scale|investigate", "target": "<service>"}',
        #     messages=[{"role": "user", "content": description}],
        # )
        # data = json.loads(raw)
        # return {"remediation": data["remediation"], "target": data["target"]}
        # --- deterministic rule (what runs) ----------------------------------
        desc = description.lower()
        remediation = "restart" if "restart" in desc else "scale" if "scale" in desc else "investigate"
        target = "unknown-target"
        for token in desc.split():
            if "-" in token and any(s in token for s in ("api", "service", "worker")):
                target = token
                break
        return {"remediation": remediation, "target": target}

    async def synthesize_answer(self, runbook: dict[str, Any]) -> str:
        """Write the knowledge answer from a retrieved runbook."""
        rb = runbook.get("runbook")
        if not rb:
            return "No matching runbook found."
        # --- LLM version (uncomment to go live) ------------------------------
        # import json
        # return await self.complete(
        #     role="knowledge",
        #     system="Write a concise triage answer for an on-call engineer, using only "
        #            "this runbook. Name the owner and list the steps.",
        #     messages=[{"role": "user", "content": json.dumps(rb)}],
        # )
        # --- deterministic rule (what runs) ----------------------------------
        steps = "; ".join(rb["steps"])
        return f"{rb['title']} (owner: {rb['owner']}). Steps: {steps}"


# Module-level singleton - re-use across requests.
_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_for_tests(stub: LLMClient | None = None) -> None:
    """Tests override the singleton with a stub so no network call happens."""
    global _client
    _client = stub
