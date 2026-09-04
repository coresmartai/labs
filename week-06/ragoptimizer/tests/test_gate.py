"""Threshold-gate tests - the gate is symmetric, its inputs are not.

No keys, no network, no model: the retriever and the LLM are stubbed. These
tests pin the design decision documented in app/main.py::_gate_channels:

    raw_ok  = top1_raw  >= similarity_threshold and spread_raw  >= spread_delta
    hyde_ok = top1_hyde >= similarity_threshold and spread_hyde >= spread_delta
    refuse unless (raw_ok or hyde_ok)

Same RULE and same VALUES as the Week 5 baseline gate; the full W6 pipeline
just gets to present two probes to it instead of one. The load-bearing test is
test_full_answers_when_raw_fails_but_hyde_clears - that is the only case where
HyDE's second channel changes the outcome, so it is the case that decides
whether HyDE earns its place in the pipeline.
"""
from __future__ import annotations

import json

import pytest

from app import main as main_mod
from app.config import get_settings
from app.llm import REFUSAL_STRING
from app.main import _gate_channels, eval_compare
from app.schemas import Chunk, Citation, GroundedAnswer, PipelineTrace

# Gate thresholds live in config; the tests read them rather than hard-coding
# 0.55/0.08, so a config change moves the tests with it.
_S = get_settings()
PASS_TOP1 = _S.similarity_threshold + 0.20     # comfortably clears
PASS_SPREAD = _S.spread_delta + 0.10
FAIL_TOP1 = _S.similarity_threshold - 0.20     # comfortably fails
FAIL_SPREAD = _S.spread_delta - 0.05


def _chunks() -> list[Chunk]:
    return [Chunk(chunk_id="c1", text="[c1]\nAttention uses scaled dot-product.", score=0.9)]


def _trace(top1_raw: float, spread_raw: float, top1_hyde: float, spread_hyde: float) -> PipelineTrace:
    return PipelineTrace(
        top1_raw=top1_raw, spread_raw=spread_raw,
        top1_hyde=top1_hyde, spread_hyde=spread_hyde,
    )


@pytest.fixture
def golden(tmp_path, monkeypatch):
    """A one-row should-answer golden set, wired into settings."""
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([{
        "question": "How does scaled dot-product attention work?",
        "expected_answer": "scaled dot-product attention",
        "must_cite": ["_answered"],
    }]), encoding="utf-8")
    monkeypatch.setattr(_S, "golden_dataset_path", str(path))
    return path


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Stub both retrievers and the LLM. Records whether the LLM was reached.

    `calls` is the proof that the gate refuses BEFORE the model is called - an
    LLM-side refusal would still show up as fallback_triggered, so counting the
    calls is what distinguishes a gate refusal from a self-reported one.
    """
    calls: list[str] = []

    def _fake_ask(query, chunks, trace=None):
        calls.append(query)
        return GroundedAnswer(
            answer="Scaled dot-product attention scales by sqrt(d_k).",
            citations=[Citation(chunk_id="c1", quote="scaled dot-product")],
            confidence="high",
            fallback_triggered=False,
            pipeline_trace=trace,
        )

    monkeypatch.setattr(main_mod, "ask_with_citations", _fake_ask)
    return calls


def _set_full(monkeypatch, trace: PipelineTrace) -> None:
    monkeypatch.setattr(
        main_mod, "retrieve_with_trace", lambda q, k=None: (_chunks(), trace)
    )


def _set_baseline(monkeypatch, top1: float, spread: float) -> None:
    monkeypatch.setattr(
        main_mod, "retrieve_baseline_gated", lambda q, k=None: (_chunks(), top1, spread)
    )


# ── The gate rule itself ──────────────────────────────────────────────────────

def test_gate_channels_reads_both_probes() -> None:
    """Each channel is judged independently against the same thresholds."""
    assert _gate_channels(_trace(PASS_TOP1, PASS_SPREAD, FAIL_TOP1, FAIL_SPREAD)) == (True, False)
    assert _gate_channels(_trace(FAIL_TOP1, FAIL_SPREAD, PASS_TOP1, PASS_SPREAD)) == (False, True)
    assert _gate_channels(_trace(PASS_TOP1, PASS_SPREAD, PASS_TOP1, PASS_SPREAD)) == (True, True)
    assert _gate_channels(_trace(FAIL_TOP1, FAIL_SPREAD, FAIL_TOP1, FAIL_SPREAD)) == (False, False)


def test_gate_requires_both_top1_and_spread() -> None:
    """A channel clears only if it passes BOTH conditions - not either."""
    # Strong top1, flat spread: everything looks alike, so the top hit is not
    # distinctively relevant. Week 5 refused this; so does Week 6.
    assert _gate_channels(_trace(PASS_TOP1, FAIL_SPREAD, 0.0, 0.0)) == (False, False)
    # Wide spread but weak top1: best hit is simply not close enough.
    assert _gate_channels(_trace(FAIL_TOP1, PASS_SPREAD, 0.0, 0.0)) == (False, False)


# ── Symmetry: W6 refuses when both channels are below threshold ───────────────

def test_full_refuses_when_both_channels_below_threshold(golden, stub_pipeline, monkeypatch) -> None:
    _set_full(monkeypatch, _trace(FAIL_TOP1, FAIL_SPREAD, FAIL_TOP1, FAIL_SPREAD))
    _set_baseline(monkeypatch, PASS_TOP1, PASS_SPREAD)

    result = eval_compare()
    row = result.rows[0]

    assert row.full_actual == "refused"
    assert row.full_fallback is True
    assert row.full_answer.startswith(REFUSAL_STRING[:30])
    assert row.full_cited_ids == []
    assert row.full_confidence == "low"
    # The LLM was called exactly once - for the baseline, which cleared its gate.
    # The full pipeline refused before reaching it.
    assert len(stub_pipeline) == 1


# ── Asymmetry of inputs: HyDE is a genuine second chance ──────────────────────

def test_full_answers_when_raw_fails_but_hyde_clears(golden, stub_pipeline, monkeypatch) -> None:
    """The case HyDE exists for - and the case that justifies the design.

    Raw query fails the gate, so the baseline refuses. The HyDE probe clears it,
    so W6 answers. Gating W6 on the raw query alone would refuse here too, by
    construction, and Week 6 would be unmeasurable on exactly the rows it was
    built to win.
    """
    _set_full(monkeypatch, _trace(FAIL_TOP1, FAIL_SPREAD, PASS_TOP1, PASS_SPREAD))
    _set_baseline(monkeypatch, FAIL_TOP1, FAIL_SPREAD)

    result = eval_compare()
    row = result.rows[0]

    assert row.baseline_actual == "refused", "baseline gates on the raw query"
    assert row.full_actual == "answered", "HyDE cleared the gate - W6 answers"
    assert row.full_fallback is False
    assert row.full_cited_ids == ["c1"]
    # Row expects an answer: the baseline takes the false refusal, W6 does not.
    assert result.baseline.false_refusal_rate == 1.0
    assert result.full.false_refusal_rate == 0.0


def test_full_answers_when_only_raw_clears(golden, stub_pipeline, monkeypatch) -> None:
    """HyDE failing is not a veto - either channel clearing is enough."""
    _set_full(monkeypatch, _trace(PASS_TOP1, PASS_SPREAD, FAIL_TOP1, FAIL_SPREAD))
    _set_baseline(monkeypatch, PASS_TOP1, PASS_SPREAD)

    row = eval_compare().rows[0]
    assert row.full_actual == "answered"
    assert row.baseline_actual == "answered"


# ── HyDE disabled / probe failed must not be a free pass ─────────────────────

def test_full_refuses_when_hyde_disabled_and_raw_fails(golden, stub_pipeline, monkeypatch) -> None:
    """HyDE off → hyde pair is (0.0, 0.0) → hyde_ok False → raw decides alone."""
    # (0.0, 0.0) is exactly what retrieve_with_trace records when HyDE is
    # disabled or the probe raises - see test_hyde_disabled_yields_zero_gate_stats.
    _set_full(monkeypatch, _trace(FAIL_TOP1, FAIL_SPREAD, 0.0, 0.0))
    _set_baseline(monkeypatch, FAIL_TOP1, FAIL_SPREAD)

    row = eval_compare().rows[0]
    assert row.full_actual == "refused"
    assert row.full_fallback is True
    assert len(stub_pipeline) == 0, "neither pipeline should reach the LLM"


def test_hyde_disabled_yields_zero_gate_stats(monkeypatch) -> None:
    """The retriever records (0.0, 0.0) for HyDE when it is off - not a pass.

    Exercises the real retrieve_with_trace; only the network-bound and
    torch-bound stages are stubbed.
    """
    from app import retriever as r

    monkeypatch.setattr(_S, "hyde_enabled", False)
    monkeypatch.setattr(r, "_hybrid_search", lambda q, k: (_chunks(), FAIL_TOP1, FAIL_SPREAD))
    monkeypatch.setattr(r, "_dense_search", lambda p, k: pytest.fail("HyDE is disabled"))
    monkeypatch.setattr(r, "rerank", lambda q, c: c)
    monkeypatch.setattr(r, "compress", lambda c, q: c)

    _chunks_out, trace = r.retrieve_with_trace("how does attention work")

    assert trace.top1_hyde == 0.0 and trace.spread_hyde == 0.0
    assert trace.top1_raw == FAIL_TOP1 and trace.spread_raw == FAIL_SPREAD
    assert _gate_channels(trace) == (False, False)


def test_hyde_gate_stats_uses_week5_formula() -> None:
    """top1 = [0], top3 = [2] (0.0 if <3), spread = max(0, top1-top3), clamped."""
    from app.retriever import hyde_gate_stats

    # Chunk.score on a dense hit IS the dense cosine, so the stats read off it.
    hits = [Chunk(chunk_id=str(i), text=f"[{i}]\nx", score=s)
            for i, s in enumerate([0.81, 0.77, 0.60])]
    assert hyde_gate_stats(hits) == (pytest.approx(0.81), pytest.approx(0.21))

    # Fewer than three hits → top3 is 0.0, so spread is just top1.
    two = [Chunk(chunk_id=str(i), text=f"[{i}]\nx", score=s) for i, s in enumerate([0.70, 0.65])]
    assert hyde_gate_stats(two) == (pytest.approx(0.70), pytest.approx(0.70))

    # Empty (probe failed) → fails the gate on its own.
    assert hyde_gate_stats([]) == (0.0, 0.0)


# ── The baseline is unchanged ─────────────────────────────────────────────────

def test_baseline_still_gates_on_raw_query(golden, stub_pipeline, monkeypatch) -> None:
    """Week 5's gate, untouched: one channel, same thresholds, refuse pre-LLM."""
    _set_full(monkeypatch, _trace(PASS_TOP1, PASS_SPREAD, PASS_TOP1, PASS_SPREAD))
    _set_baseline(monkeypatch, FAIL_TOP1, FAIL_SPREAD)

    row = eval_compare().rows[0]
    assert row.baseline_actual == "refused"
    assert row.baseline_fallback is True
    assert row.baseline_answer.startswith(REFUSAL_STRING[:30])
    assert row.baseline_cited_ids == []
    # Only the full pipeline reached the LLM.
    assert len(stub_pipeline) == 1


def test_baseline_answers_when_gate_clears(golden, stub_pipeline, monkeypatch) -> None:
    _set_full(monkeypatch, _trace(PASS_TOP1, PASS_SPREAD, PASS_TOP1, PASS_SPREAD))
    _set_baseline(monkeypatch, PASS_TOP1, PASS_SPREAD)

    row = eval_compare().rows[0]
    assert row.baseline_actual == "answered"
    assert row.baseline_cited_ids == ["c1"]
    assert len(stub_pipeline) == 2, "both pipelines cleared their gates"
