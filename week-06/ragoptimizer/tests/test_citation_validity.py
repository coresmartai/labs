"""The metric must be able to say no.

Week 5 caught a citation metric that was True for every possible input, and
built a rule out of it: for every metric you report, ask what input would make
this number drop. If nothing can, it is a decoration, not a measurement.

The first draft of this week's `_judge_row` shipped that same bug back in, as
`cited_ids.issubset(must_cite_set | cited_ids)`. These tests are the guard that
stops it returning a third time. They exercise the judge directly, with no
model call and no index.
"""
from __future__ import annotations

from app.main import _judge_row
from app.schemas import Citation, GoldenRow, GroundedAnswer

ANSWERABLE = GoldenRow(
    question="What is scaled dot-product attention?",
    expected_answer="attention weights computed from queries keys values",
    must_cite=["_answered"],
)
OFF_TOPIC = GoldenRow(
    question="What is the lunchroom Wi-Fi password?",
    expected_answer="",
    must_cite=[],
)


def _answer(**kw) -> GroundedAnswer:
    base = dict(
        answer="Attention weights computed from queries keys and values.",
        citations=[Citation(chunk_id="doc#1", quote="q")],
        confidence="high",
        fallback_triggered=False,
        generated=True,
        citations_dropped=0,
    )
    base.update(kw)
    return GroundedAnswer(**base)


def test_the_old_expression_could_never_fail() -> None:
    """Documents the bug this file exists to prevent."""
    for cited, must in (
        (set(), set()),
        ({"doc#1"}, set()),
        ({"doc#9"}, {"doc#1"}),          # a fabricated citation
        ({"doc#1", "doc#7"}, {"doc#2"}),
    ):
        assert cited.issubset(must | cited) is True, "should be vacuously true"


def test_clean_answer_scores_valid() -> None:
    scores = _judge_row(ANSWERABLE, _answer())
    assert scores["cit_valid_ok"] is True


def test_fabricated_citation_scores_invalid() -> None:
    """The metric says no. This is the case the old expression could not see."""
    scores = _judge_row(ANSWERABLE, _answer(citations_dropped=1))
    assert scores["cit_valid_ok"] is False


def test_gate_refusal_is_excluded_not_passed() -> None:
    """A row the gate refused is not evidence of good citation behaviour."""
    refused = _answer(
        answer="I don't have that information in the provided sources.",
        citations=[], fallback_triggered=True, generated=False,
    )
    assert _judge_row(ANSWERABLE, refused)["cit_valid_ok"] is None
    assert _judge_row(OFF_TOPIC, refused)["cit_valid_ok"] is None


def test_correct_refusal_of_off_topic_still_excluded() -> None:
    """Right outcome, but citation discipline was never exercised."""
    refused = _answer(citations=[], fallback_triggered=True, generated=False)
    scores = _judge_row(OFF_TOPIC, refused)
    assert scores["false_answer"] is False
    assert scores["cit_valid_ok"] is None


def test_metric_is_not_an_alias_for_false_answer_rate() -> None:
    """The old implementation moved only when false_answer moved. This does not."""
    fabricated = _judge_row(ANSWERABLE, _answer(citations_dropped=2))
    assert fabricated["cit_valid_ok"] is False
    assert fabricated["false_answer"] is False   # the answer was still grounded
