"""The research loop. One question at a time, over the corpus, into the ledger.

Structurally this is OpsAssist's loop with three things changed: the tools, the
stop conditions, and what the run produces. The four states are the same, the
guards at the top are the same, and the trace is the same shape.

Read `run_question` beside OpsAssist's `run_agent` and the diff is the project.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.config import get_settings
from app.ledger import LedgerProtocol
from app.llm import ModelResponse, call_with_tools
from app.retrieval import CorpusIndex
from app.schemas import LedgerEntry, TraceIteration
from app.tools import catalog_for_model, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research assistant working over a fixed corpus of IETF RFCs about \
email authentication: SPF, DKIM, DMARC, ARC and Authentication-Results.

You have three tools: search_corpus, read_chunk and list_documents. Their descriptions say \
when to use each.

How to work:
- Search before you claim. A claim you cannot point at a passage for is a claim you do not make.
- Search more than once. The first query rarely settles a question, and different wordings \
surface different passages.
- Read a chunk in full before quoting from it.

What you must not do:
- Do not answer from what you already know about these protocols. The corpus is the only \
source. Your training data may be older than these documents, and where they disagree, the \
corpus is right.
- Do not invent a chunk_uid. Use only ones a search returned.

Stopping:
- When you have enough evidence to answer, say so and stop.
- When the corpus does not contain the answer, SAY THAT AND STOP. Declining is a correct \
outcome and searching ten times for something that is not there is not. If two or three \
different searches have surfaced nothing relevant, the answer is that the corpus does not \
cover it."""


def run_question(
    index: CorpusIndex,
    ledger: LedgerProtocol,
    tools: dict[str, dict[str, Any]],
    question: str,
    question_index: int,
) -> tuple[list[TraceIteration], str, list[LedgerEntry], str]:
    """Research one question. Returns (trace, stop_reason, evidence, final_text)."""
    settings = get_settings()
    catalog = catalog_for_model(tools)
    trace: list[TraceIteration] = []
    stop_reason = "max_iters"
    started = time.monotonic()
    tokens_used = 0
    cost_usd = 0.0
    final_text = ""
    evidence: list[LedgerEntry] = []

    messages: list[dict] = [{"role": "user", "content": question}]

    for it in range(settings.max_iters):
        iter_start = time.time()

        # Guard 1: wall clock.
        if time.monotonic() - started >= settings.max_wall_clock_seconds:
            stop_reason = "timeout"
            break

        # Guard 2: token budget.
        if tokens_used > settings.token_budget:
            stop_reason = "token_budget"
            break

        # Guard 3: the dollar ceiling. Off unless MAX_COST_USD is set.
        if settings.max_cost_usd is not None and cost_usd >= settings.max_cost_usd:
            stop_reason = "cost_budget"
            break

        # -----------------------------------------------------------------
        # Guard 4: YOUR STOP CONDITION.
        #
        # Nothing above answers "is this research finished?". Add the guard
        # that does, and give it a stop_reason of its own so the trace names
        # it. See app/config.py for where its settings belong.
        # -----------------------------------------------------------------

        try:
            resp: ModelResponse = call_with_tools(
                system_prompt=SYSTEM_PROMPT, messages=messages, tools=catalog)
        except Exception as exc:  # noqa: BLE001
            stop_reason = "fatal_tool_error"
            final_text = f"Model call failed: {exc}"
            break

        tokens_used += resp.input_tokens + resp.output_tokens
        cached = min(max(resp.cached_input_tokens, 0), resp.input_tokens)
        cost_usd += (
            (resp.input_tokens - cached) / 1e6 * settings.price_per_1m_input_usd
            + cached / 1e6 * settings.price_per_1m_cached_input_usd
            + resp.output_tokens / 1e6 * settings.price_per_1m_output_usd
        )
        tool_blocks = [b for b in resp.content_blocks if b["type"] == "tool_call"]
        text_blocks = [b for b in resp.content_blocks if b["type"] == "text"]

        if resp.stop_reason == "end_turn" or not tool_blocks:
            final_text = "\n".join(t["text"] for t in text_blocks)
            stop_reason = "end_turn"
            trace.append(TraceIteration(
                iter=it, question_index=question_index,
                model_request_tokens=resp.input_tokens,
                model_response_tokens=resp.output_tokens,
                tool_calls=[], tool_results=[], new_ledger_entries=0,
                elapsed_ms=int((time.time() - iter_start) * 1000)))
            break

        messages.append({
            "role": "assistant",
            "content": "\n".join(t["text"] for t in text_blocks) if text_blocks else None,
            "tool_calls": [{"id": b["id"], "type": "function",
                            "function": {"name": b["name"],
                                         "arguments": json.dumps(b["input"])}}
                           for b in tool_blocks],
        })

        calls, results, new_entries = [], [], 0
        for block in tool_blocks:
            output = execute_tool(tools, block["name"], block["input"])
            calls.append({"name": block["name"], "input": block["input"], "id": block["id"]})

            # The crossing that matters here: a search's hits go into the ledger
            # BEFORE the model sees them, so every passage the model could cite
            # already has an identity. Nothing downstream has to guess.
            if block["name"] == "search_corpus" and output.get("success"):
                hits = index.search(block["input"]["query"],
                                    k=int(block["input"].get("k", 5)))
                before = len(ledger.entries())
                recorded = ledger.record(hits, block["input"]["query"], it, question_index)
                new_entries += len(ledger.entries()) - before
                evidence.extend(recorded)
                # Give the model the ledger ids, not the ranks. This is the line
                # that makes citation possible at all.
                for hit, entry in zip(hits, recorded):
                    for h in output["hits"]:
                        if h["chunk_uid"] == hit.chunk.chunk_uid:
                            h["ledger_id"] = entry.ledger_id

            # Serialised AFTER the ledger ids are attached, so the trace records
            # what the model actually saw rather than what the tool first returned.
            results.append({"tool_call_id": block["id"],
                            "is_error": output.get("success") is False,
                            "content_preview": json.dumps(output)[:220]})
            messages.append({"role": "tool", "tool_call_id": block["id"],
                             "content": json.dumps(output)})

            if output.get("success") is False and output.get("terminal"):
                stop_reason = "fatal_tool_error"

        trace.append(TraceIteration(
            iter=it, question_index=question_index,
            model_request_tokens=resp.input_tokens,
            model_response_tokens=resp.output_tokens,
            tool_calls=calls, tool_results=results,
            new_ledger_entries=new_entries,
            elapsed_ms=int((time.time() - iter_start) * 1000)))

        if stop_reason == "fatal_tool_error":
            break

    return trace, stop_reason, evidence, final_text
