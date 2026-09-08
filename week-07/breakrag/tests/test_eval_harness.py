"""The live eval suite - run with a real SUT and real judge API keys.

Run with:
    pytest -m live tests/test_eval_harness.py -v

The four assertions below are the thresholds that define a passing run. The
leakage check comes first on purpose: a score earned on contaminated data is
not a score, so the other three are only worth reading when it passes.
Each has a custom error message so a failure at the wrong hour names the
problem and points to the first place to look.

Note: this suite reaches out to the real SUT and real judge APIs.
Mark `pytest.mark.live` keeps it out of the fast offline smoke suite.
Run `pytest tests/test_smoke.py -v` before pushing; run this only when
you have a live SUT and OPENAI_API_KEY configured.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.adversarial import expand, load_seeds
from app.config import get_settings
from app.judges import available_judges, score_with_all
from app.leakage import collect_leakage_inputs, probe_canaries
from app.scorecard import build_scorecard, render_markdown, write_markdown
from app.schemas import CaseResult
from app.system_under_test import call_sut, SystemUnderTestError


pytestmark = pytest.mark.live


def _judges_or_fail():
    """Return configured judges, or fail early with a clear message."""
    judges = available_judges()
    assert len(judges) >= 1, (
        f"Live eval needs at least one judge; got {len(judges)}. "
        "Set OPENAI_API_KEY in .env and restart the process."
    )
    return judges


async def _run_suite() -> tuple[list[CaseResult], list]:
    seeds = load_seeds()
    cases = expand(seeds)
    judges = _judges_or_fail()
    results: list[CaseResult] = []
    for case in cases:
        try:
            response = await call_sut(case)
        except SystemUnderTestError as exc:
            pytest.fail(f"SUT call failed on {case.id}: {exc}")
            return [], []
        verdicts = await score_with_all(judges, case, response)
        results.append(
            CaseResult(case=case, response=response, judge_verdicts=verdicts)
        )
    return results, seeds


@pytest.fixture(scope="session")
def suite_results():
    results, seeds = asyncio.run(_run_suite())
    # The same inputs the server uses for /run: the indexed passages are the
    # haystack, their vectors the baseline, and every canary is probed.
    inputs = collect_leakage_inputs(seeds)
    canaries = asyncio.run(probe_canaries(seeds))
    sc = build_scorecard(
        results,
        seeds=seeds,
        leakage_haystack=inputs.haystack,
        seed_embeddings=inputs.seed_embeddings,
        corpus_embeddings=inputs.corpus_embeddings,
        canary_completions=canaries,
        leakage_inputs_note=inputs.note,
    )
    # Write the scorecard markdown so you can review it after the run.
    out_path = Path("scorecards/last.md")
    out_path.parent.mkdir(exist_ok=True)
    write_markdown(sc, out_path)
    return results, sc


def test_leakage_check(suite_results):
    """All three leakage checks must pass before any other score counts.

    n-gram (verbatim), embedding similarity (near-duplicate, calibrated against
    the corpus's own neighbour distribution), and canary (memorisation). A score
    earned on contaminated data is not a score.
    """
    _, sc = suite_results
    assert sc.leakage_check_status != "skipped", (
        f"Leakage check did not run: {sc.leakage_check_notes}. "
        "A skipped check is not a pass. Check QDRANT_URL and OPENAI_API_KEY."
    )
    assert sc.leakage_check_passed, (
        f"Leakage check FAILED: {sc.leakage_check_notes}. "
        "Rotate affected canaries and rebuild the contaminated portion of the golden set."
    )


def test_min_sample_size(suite_results):
    """A confident number on n < 100 is fiction - that's the production rule.

    The floor is set to 50 here because the classroom run is 70 cases
    (10 seeds × 7 cases/seed); raise MIN_SAMPLES to 100+ in production.
    """
    results, _ = suite_results
    settings = get_settings()
    assert len(results) >= settings.min_samples, (
        f"BreakRAG suite produced only {len(results)} cases; "
        f"minimum is {settings.min_samples}. "
        "Refusing to report a fake-confident score on a tiny sample."
    )


def test_adversarial_pass_rate(suite_results):
    """Refuse/match/abstain behaviour must hold across all adversarial strategies."""
    _, sc = suite_results
    settings = get_settings()
    floor = settings.adversarial_pass_rate_floor
    bad_strategies: list[str] = []
    for row in sc.rows:
        if row.metric.startswith("adv_pass_rate["):
            if row.current < floor:
                bad_strategies.append(f"{row.metric}={row.current:.3f}")
    assert not bad_strategies, (
        f"Adversarial pass-rate floor {floor:.2f} breached on: "
        + ", ".join(bad_strategies)
        + ". Likely a guardrails regression - inspect refusal cases first."
    )


def test_judge_agreement_alpha(suite_results):
    """Krippendorff's alpha: if the judges can't agree, the score is fiction."""
    _, sc = suite_results
    settings = get_settings()
    alpha = sc.judge_agreement_alpha
    assert alpha >= settings.judge_agreement_alpha_floor, (
        f"Judge agreement alpha = {alpha:.3f} below floor "
        f"{settings.judge_agreement_alpha_floor:.2f}. "
        "Roll back the most recent rubric change and re-run the calibration set."
    )
