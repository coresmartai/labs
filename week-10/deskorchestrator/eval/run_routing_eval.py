"""TASK 2's harness. This ships working; the labelled set is yours.

Scores the ROUTER IN ISOLATION: it calls the classifier directly and never runs a
specialist, so no worker can influence the number. That isolation is the step
people skip, and it is what makes a regression in answers trace cleanly to either
the supervisor or a worker rather than to the soup of both.

It uses the bare triage prompt rather than the composed one, so the number is not
moved by one user's procedural preferences. Production composes; this measures.

    python eval/run_routing_eval.py            # needs a real key; one nano call per case
    python eval/run_routing_eval.py --dry-run  # no calls; checks your file parses
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm                                    # noqa: E402
from app.config import get_settings                    # noqa: E402
from app.graph import ROUTES                           # noqa: E402

SET_PATH = Path(__file__).parent / "routing_eval.json"


def load() -> list[dict]:
    cases = json.loads(SET_PATH.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        sys.exit("routing_eval.json must be a JSON list of cases")
    for i, c in enumerate(cases):
        if not {"request", "expected"} <= set(c):
            sys.exit(f"case {i} needs both 'request' and 'expected'. Got keys: {sorted(c)}")
        if c["expected"] not in ROUTES:
            sys.exit(f"case {i}: expected {c['expected']!r} is not one of {ROUTES}")
    return cases


def classify(request: str) -> str:
    from prompts import TRIAGE_BASE
    s = get_settings()
    r = llm.call(s.triage_model, TRIAGE_BASE, request, max_tokens=20)
    route = r.text.strip().lower().strip(' ".')
    return route if route in ROUTES else "escalate_human"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="parse and report, make no calls")
    args = ap.parse_args()

    cases = load()
    dist = Counter(c["expected"] for c in cases)
    print(f"{len(cases)} cases. Expected-route distribution: {dict(dist)}")
    if len(cases) < 20:
        print(f"  NOTE: {len(cases)} cases. Twenty is the usual floor; effect sizes here are large "
              "enough that a small set still moves, but fewer than twenty is hard to defend.")
    missing = [r for r in ROUTES if r not in dist]
    if missing:
        print(f"  NOTE: no case expects {missing}. A route you never test is a route you never measure.")
    if args.dry_run:
        return 0

    confusion: Counter = Counter()
    correct = 0
    for c in cases:
        got = classify(c["request"])
        ok = got == c["expected"]
        correct += ok
        if not ok:
            confusion[(c["expected"], got)] += 1
            print(f"  MISS  expected {c['expected']:<15} got {got:<15} {c['request'][:58]}")

    pct = 100.0 * correct / len(cases)
    print(f"\nrouting accuracy: {correct}/{len(cases)} = {pct:.1f}%")
    if confusion:
        print("confusions, worst first:")
        for (exp, got), n in confusion.most_common():
            print(f"  {n:>3}x  {exp} -> {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
