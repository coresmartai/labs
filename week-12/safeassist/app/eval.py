"""Eval harness - the security layer's smoke test, scored.

A plain RAG eval asks: is the answer grounded, does it cite, does it refuse
when it should. This is a *hardened* service, so the eval scores the security
behaviours too: does an injection get refused, does the ACL actually hide a
chunk, does PII ever reach the output. Each golden row declares the behaviour
it expects; `evaluate()` runs the real pipeline and checks.

This runs the SAME code path as `POST /ask` - `app.pipeline` - so a green eval
means the live route behaves as claimed, not that a parallel re-implementation
does. It calls the real model and a real Qdrant; tests stub both, so `pytest`
scores the harness offline.

The golden set is hand-authored (`data/golden_eval.json`) rather than
LLM-generated: the corpus is twenty known chunks with known ACLs, so the
answerable / refusable / injection / ACL cases can be written by hand and stay
deterministic. No build step, no generation cost.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import get_settings
from app.injection import classify
from app.pipeline import generate_sentences, retrieve
from app.presidio_layer import scrub_input
from app.schemas import EvalCase, EvalCaseResult, EvalReport

logger = logging.getLogger(__name__)

_GOLDEN_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_eval.json"


def load_golden() -> list[EvalCase]:
    """Read the hand-authored golden rows from data/golden_eval.json."""
    rows = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    return [EvalCase.model_validate(r) for r in rows]


def run_case(case: EvalCase) -> EvalCaseResult:
    """Run one row through the real pipeline and record what happened.

    Mirrors `app.main.ask` step for step - input scrub, injection classifier,
    retrieval (ACL + floor + sanitiser), generation - but headless: no SSE, no
    HTTP, and therefore no RBAC decorator (RBAC is an endpoint concern, tested
    against the route itself). Roles come from the row.
    """
    events: list[str] = []

    def log(event_type: str, detail: dict, action_taken: str) -> None:
        events.append(event_type)

    # 1. Input scrub (query PII never reaches the model or a log).
    scrubbed = scrub_input(case.question)

    # 2. Injection classifier - the route refuses outright when it fires.
    settings = get_settings()
    if settings.injection_classifier_enabled and classify(case.question).flagged:
        return _result(case, actual="injection_refusal", cited=[], events=["injection_refusal"],
                       answer="", leaked=False)

    # 3. Retrieval (ACL pushed down, relevance floor, injection sanitiser).
    chunks = retrieve(scrubbed.cleaned_text, case.roles, log)
    if not chunks:
        # No relevant, visible context. If the ACL logged a withheld chunk the
        # question WAS answerable for someone - that is an acl_block, not a
        # plain off-topic refusal.
        actual = "acl_block" if "acl_filter" in events else "refuse"
        return _result(case, actual=actual, cited=[], events=events, answer="", leaked=False)

    # 4. Generate (real model, streamed through the output scrubber, then joined).
    answer = "".join(generate_sentences(scrubbed.cleaned_text, chunks))
    cited = [c.chunk_id for c in chunks]
    quotes = " ".join(c.text for c in chunks)
    leaked = any(s in answer or s in quotes for s in case.must_not_leak)
    return _result(case, actual="answer", cited=cited, events=events, answer=answer, leaked=leaked)


def _result(case: EvalCase, *, actual: str, cited: list[str], events: list[str],
            answer: str, leaked: bool) -> EvalCaseResult:
    return EvalCaseResult(
        question=case.question,
        roles=case.roles,
        expected=case.expected,
        actual=actual,
        passed=(actual == case.expected) and not leaked,
        cited_ids=cited,
        events=events,
        leaked=leaked,
        answer_preview=answer[:160],
    )


def evaluate(cases: list[EvalCase] | None = None) -> EvalReport:
    """Run every golden case and aggregate. `cases=None` loads the golden set."""
    cases = cases if cases is not None else load_golden()
    rows = [run_case(c) for c in cases]

    by: dict[str, dict[str, int]] = {}
    for r in rows:
        b = by.setdefault(r.expected, {"total": 0, "correct": 0})
        b["total"] += 1
        b["correct"] += int(r.passed)

    correct = sum(int(r.passed) for r in rows)
    leaks = sum(int(r.leaked) for r in rows)
    total = len(rows)
    return EvalReport(
        total=total,
        correct=correct,
        accuracy=round(correct / total, 4) if total else 0.0,
        pii_leaks=leaks,
        by_expected=by,
        rows=rows,
    )
