"""Replay a saved OpsAssist trace - pretty-print the iteration timeline.

The trace is a first-class return value: it records every model decision,
tool call, and tool result, so a failing production run can be inspected
locally with zero API calls and zero spend.

Usage:
    python replay.py trace.json

`trace.json` is a saved AgentResponse body, e.g.:
    curl -s -X POST http://localhost:8000/agent/run ... > trace.json
(A bare trace array works too.)
"""
from __future__ import annotations

import json
import sys


def load(path: str) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):                      # bare trace array
        return {}, data
    return data, data.get("trace", [])


def print_iteration(it: dict) -> None:
    tokens = it.get("model_request_tokens", 0) + it.get("model_response_tokens", 0)
    print(f"--- iter {it.get('iter', '?')} "
          f"({tokens} tok, {it.get('elapsed_ms', '?')} ms) ---")
    calls = it.get("tool_calls", [])
    if not calls:
        print("    (no tool calls - final turn)")
    for call in calls:
        args = json.dumps(call.get("input", {}))
        print(f"    -> {call.get('name')}({args})  id={call.get('id')}")
    for res in it.get("tool_results", []):
        flag = "ERROR" if res.get("is_error") else "ok"
        preview = res.get("content_preview", "")[:80]
        print(f"    <- [{flag}] tool_call_id={res.get('tool_call_id')}  {preview}")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    envelope, trace = load(sys.argv[1])

    print("=" * 64)
    print("OpsAssist trace replay")
    if envelope:
        print(f"stop_reason : {envelope.get('stop_reason')}")
        print(f"iter_count  : {envelope.get('iter_count')}")
    print(f"trace steps : {len(trace)}")
    print("=" * 64)

    total_tokens = 0
    for it in trace:
        print_iteration(it)
        total_tokens += (it.get("model_request_tokens", 0)
                         + it.get("model_response_tokens", 0))

    print("=" * 64)
    print(f"total tokens: {total_tokens}")
    if envelope.get("final_response"):
        print("final response:")
        print(f"  {envelope['final_response']}")


if __name__ == "__main__":
    main()
