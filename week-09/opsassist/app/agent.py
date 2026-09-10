"""The four-step agent loop from V1, in Python (OpenAI function-calling shape).

session start -> load_session_context()   # the ONE persistent read (crossing 3)

while iter_count < MAX_ITERS:
    0. Loop guards      -> wall-clock timeout, cached consent flag,
                           token budget, dollar-cost ceiling
    1. Model decides    -> call_with_tools(...)
    2. Tool executes    -> execute_tool(...)
    3. Result injected  -> role:tool message with the correct tool_call_id
                           -> checkpoint after every successful tool call
    4. Stop check       -> end_turn, max_iters, token_budget, cost_budget,
                           fatal_tool_error, consent_revoked, timeout

session end   -> append_audit()            # the ONE persistent write

Zero persistent reads happen inside the loop.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from app.config import get_settings
from app.llm import call_with_tools, ModelResponse
from app.state import Session
from app.tools import execute_tool, tool_catalog_for_model
from app.schemas import AgentResponse, TraceIteration

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an ops assistant. You help engineers diagnose service issues by consulting runbooks, querying metrics, and proposing remediations.

You have three tools:
- Use `get_runbook` to look up the canonical procedure for a known issue type.
- Use `query_metrics` only after you have a specific service and time window in mind. Do not query speculatively.
- Use `propose_remediation` only after you have evidence from at least one runbook and one metric query. Always supply an idempotency_key (a UUID-style string).

Stopping policy:
If you cannot answer with the tools available, return a final response that says so and stop. Do not invent tools that do not exist. Do not retry a failing tool more than twice. If the situation requires human judgement, propose escalation and stop.
"""


def run_agent(session: Session, user_input: str) -> AgentResponse:
    """The agent loop.

    The ~180 lines that fuse V1, V2, V3 into a running program.
    """
    settings = get_settings()
    catalog = tool_catalog_for_model()
    trace: list[TraceIteration] = []
    stop_reason = "max_iters"
    run_started = time.monotonic()

    # Crossing 3: the single persistent read of the whole run - consent and
    # user preferences together, in one query, at session start. Both are
    # cached on the Session; the loop never touches the database again.
    ctx = session.persistent.load_session_context(session.user_id)
    session.consent = ctx["consent"]
    session.prefs = ctx["prefs"]

    if not session.consent:
        return AgentResponse(
            final_response="I cannot proceed - consent has been revoked.",
            iter_count=0,
            stop_reason="consent_revoked",
            trace=[],
        )

    # The prefs we just read are not decoration: they go into the prompt.
    prompt = SYSTEM_PROMPT
    if session.prefs:
        prompt += f"\n\nUser preferences in force: {json.dumps(session.prefs)}"

    # Seed the conversation (a resumed run already carries earlier turns).
    session.short_term.append_message({"role": "user", "content": user_input})

    final_text = ""

    while session.short_term.iter_count < settings.max_iters:
        iter_index = session.short_term.iter_count
        iter_start = time.time()

        # Stop condition: wall-clock timeout.
        if time.monotonic() - run_started >= settings.max_wall_clock_seconds:
            stop_reason = "timeout"
            break

        # Stop condition: consent revoked. Checked against the flag cached at
        # session start - an in-process read, NOT a persistent-layer crossing.
        # The honest trade: a revocation lands on the next request, not
        # mid-loop. That is the price of a clean boundary.
        if not session.consent:
            stop_reason = "consent_revoked"
            final_text = "I cannot proceed - consent has been revoked."
            break

        # Stop condition: token budget.
        if session.short_term.remaining_budget() < 500:
            stop_reason = "token_budget"
            break

        # Stop condition: dollar-cost ceiling. Off unless MAX_COST_USD is set.
        if (settings.max_cost_usd is not None
                and session.short_term.cost_usd >= settings.max_cost_usd):
            stop_reason = "cost_budget"
            final_text = (
                f"Stopped: cost ceiling of ${settings.max_cost_usd:.4f} reached "
                f"(spent ${session.short_term.cost_usd:.4f})."
            )
            break

        # Step 1: model decides.
        try:
            resp: ModelResponse = call_with_tools(
                system_prompt=prompt,
                messages=session.short_term.messages,
                tools=catalog,
            )
        except Exception as exc:  # noqa: BLE001 - we want any LLM error to abort
            stop_reason = "fatal_tool_error"
            final_text = f"Model call failed: {exc}"
            break

        session.short_term.record_usage(
            resp.input_tokens, resp.output_tokens, resp.cached_input_tokens
        )

        # Step 4 (early exit): end_turn -> task complete.
        tool_call_blocks = [b for b in resp.content_blocks if b["type"] == "tool_call"]
        text_blocks = [b for b in resp.content_blocks if b["type"] == "text"]

        if resp.stop_reason == "end_turn" or not tool_call_blocks:
            final_text = "\n".join(t["text"] for t in text_blocks)
            stop_reason = "end_turn"
            # Record this final iteration in the trace.
            trace.append(TraceIteration(
                iter=iter_index,
                model_request_tokens=resp.input_tokens,
                model_response_tokens=resp.output_tokens,
                tool_calls=[],
                tool_results=[],
                elapsed_ms=int((time.time() - iter_start) * 1000),
            ))
            break

        # Append the assistant turn in OpenAI format: text content + tool_calls list.
        session.short_term.append_message({
            "role": "assistant",
            "content": "\n".join(t["text"] for t in text_blocks) if text_blocks else None,
            "tool_calls": [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        "arguments": json.dumps(b["input"]),
                    },
                }
                for b in tool_call_blocks
            ],
        })

        # Steps 2 + 3: execute every tool the model asked for, inject each
        # result as a separate role:tool message (the id binding that matters).
        tool_call_records: list[dict] = []
        tool_result_records: list[dict] = []

        for block in tool_call_blocks:
            output = execute_tool(block["name"], block["input"])
            result_str = json.dumps(output)
            is_error = output.get("success") is False
            if is_error and output.get("error") == "unknown_tool":
                stop_reason = "fatal_tool_error"

            tool_call_records.append({
                "name": block["name"], "input": block["input"], "id": block["id"]
            })
            tool_result_records.append({
                "tool_call_id": block["id"], "is_error": is_error,
                "content_preview": result_str[:120],
            })

            # Step 3: one role:tool message per call (OpenAI shape).
            session.short_term.append_message({
                "role": "tool",
                "tool_call_id": block["id"],    # <-- the id binding that matters
                "content": result_str,
            })

            # Crossing 2: a checkpoint after every SUCCESSFUL tool call.
            # A recovery hint, not a full snapshot - task id, current sub-step,
            # awaiting-input flag, last successful tool-call id, a one-line
            # digest and a pointer to the transcript. No "messages" key: the
            # conversation never goes into Redis. The payload is a few hundred
            # bytes, which is what makes a per-tool-call write affordable.
            if not is_error:
                session.workflow.write_checkpoint({
                    "task_id": session.task_id,
                    "iter": session.short_term.iter_count,
                    "awaiting_input": False,
                    "last_tool_call_id": block["id"],
                    "summary": session.short_term.digest(),
                    "transcript_ref": f"audit:{session.user_id}:{session.task_id}",
                    "tokens_used": session.short_term.tokens_used,
                    "ts": time.time(),
                })

        trace.append(TraceIteration(
            iter=iter_index,
            model_request_tokens=resp.input_tokens,
            model_response_tokens=resp.output_tokens,
            tool_calls=tool_call_records,
            tool_results=tool_result_records,
            elapsed_ms=int((time.time() - iter_start) * 1000),
        ))

        session.short_term.iter_count += 1

        if stop_reason == "fatal_tool_error":
            break

    # Crossing 3, second half: the single persistent write, at session end.
    # The transcript goes here - it is what `transcript_ref` on the checkpoint
    # points at, and what `rehydrate_full_transcript` rebuilds a run from.
    session.persistent.append_audit(session.user_id, {
        "task_id": session.task_id,
        "iter_count": session.short_term.iter_count,
        "stop_reason": stop_reason,
        "transcript": session.short_term.messages,
    })

    return AgentResponse(
        final_response=final_text or "(no final response produced)",
        iter_count=session.short_term.iter_count,
        stop_reason=stop_reason,
        trace=trace,
    )


def new_task_id() -> str:
    return str(uuid.uuid4())
