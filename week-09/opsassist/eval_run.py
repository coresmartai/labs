"""Five-scenario golden set for OpsAssist (the V4 eval hook).

Runs `run_agent` in-process against a scripted stub model - no API key
needed, no spend. Three scenarios are expected to pass; two are expected
to fail, matching the on-camera demo: one wrong remediation, one missing
`escalate_to_human` tool.

Usage: python eval_run.py
"""
from __future__ import annotations

import os
import uuid

os.environ.setdefault("OPENAI_API_KEY", "eval-stub-not-a-real-credential")

from app import agent as agent_module           # noqa: E402
from app.agent import run_agent                 # noqa: E402
from app.llm import ModelResponse               # noqa: E402
from app.state import (                         # noqa: E402
    PersistentState, Session, ShortTermState, WorkflowState,
)


def _tool(name: str, args: dict) -> ModelResponse:
    return ModelResponse(
        content_blocks=[{"type": "tool_call", "id": f"call_{uuid.uuid4().hex[:8]}",
                         "name": name, "input": args}],
        stop_reason="tool_calls", input_tokens=900, output_tokens=120)


def _final(text: str) -> ModelResponse:
    return ModelResponse(content_blocks=[{"type": "text", "text": text}],
                         stop_reason="end_turn", input_tokens=900, output_tokens=80)


def _key() -> str:
    return str(uuid.uuid4())


# Golden set: (name, prompt, scripted model turns, expected keyword in final text).
SCENARIOS = [
    ("auth-latency", "Investigate auth-service latency and propose a remediation.",
     [_tool("get_runbook", {"topic": "auth-service-latency"}),
      _tool("query_metrics", {"service": "auth-service", "window": "1h"}),
      _tool("propose_remediation", {"issue": "latency-spike", "severity": "high", "idempotency_key": _key()}),
      _final("Latency spike confirmed. Recommend rotate-credentials.")], "rotate-credentials"),
    ("api-error-spike", "Error rate on the API is elevated. Diagnose it.",
     [_tool("get_runbook", {"topic": "api-error-spike"}),
      _tool("query_metrics", {"service": "api", "window": "5m"}),
      _tool("propose_remediation", {"issue": "error-spike", "severity": "high", "idempotency_key": _key()}),
      _final("Error spike confirmed at 4.2%. Recommend rollback-deploy.")], "rollback-deploy"),
    ("queue-backlog", "Worker queue backlog is growing. Propose a scale action.",
     [_tool("get_runbook", {"topic": "worker-queue-backlog"}),
      _tool("query_metrics", {"service": "worker", "window": "1h"}),
      _tool("propose_remediation", {"issue": "queue-backlog", "severity": "medium", "idempotency_key": _key()}),
      _final("Backlog confirmed. Recommend scale-worker-pool.")], "scale-worker-pool"),
    ("wrong-remediation", "auth-service latency is up. What should we do?",
     [_tool("get_runbook", {"topic": "auth-service-latency"}),
      _tool("query_metrics", {"service": "auth-service", "window": "1h"}),
      _final("Recommend restart-pod for auth-service.")], "rotate-credentials"),
    ("needs-escalation", "Escalate this incident to the on-call human immediately.",
     [_tool("escalate_to_human", {"reason": "user requested escalation"}),
      _final("Escalated.")], "escalated to on-call"),
]


def run_scenario(name, prompt, script, expected_keyword):
    turns = iter(script)
    agent_module.call_with_tools = lambda **kw: next(turns)
    task_id = f"eval-{name}"
    session = Session(task_id=task_id, user_id="eval",
                      short_term=ShortTermState(),
                      workflow=WorkflowState(task_id=task_id),
                      persistent=PersistentState())
    resp = run_agent(session, prompt)
    if resp.stop_reason != "end_turn":
        return False, f"stop_reason={resp.stop_reason}"
    if expected_keyword.lower() not in resp.final_response.lower():
        return False, f"expected '{expected_keyword}' in final response"
    return True, f"end_turn in {len(resp.trace)} iterations"


def main() -> None:
    print(f"{'Scenario':<20} {'Result':<6} Detail")
    print("-" * 64)
    passed = 0
    for name, prompt, script, expected in SCENARIOS:
        ok, detail = run_scenario(name, prompt, script, expected)
        passed += ok
        print(f"{name:<20} {'PASS' if ok else 'FAIL':<6} {detail}")
    print("-" * 64)
    print(f"{passed}/{len(SCENARIOS)} scenarios passed "
          f"(expected: 3 pass, 2 fail - see the V4 eval-hook beat)")


if __name__ == "__main__":
    main()
