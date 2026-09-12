"""Measuring the detector on your own data.

Reading 2 gave you published figures for this class of detector on real text:
recall between 0.71 and 0.94, precision between 0.15 and 0.51. The spread
between two domains is more than three-fold, which is the reason this file
exists. You cannot import somebody else's numbers. You have to measure yours.

WHAT IS MEASURED

A labelled case is a piece of text plus the spans in it that a human says are
personal data. Running the analyzer over the text gives predicted spans. Then:

    true positive   a predicted span overlaps a labelled span of the same type
    false positive  a predicted span overlaps nothing labelled
    false negative  a labelled span nothing predicted overlaps

    precision = TP / (TP + FP)      of everything we redacted, how much was PII
    recall    = TP / (TP + FN)      of the PII present, how much did we catch

Overlap rather than exact match, deliberately. A detector that finds
"Alex Whitfield" where the label says "Alex Whitfield " differs by a trailing
space, and counting that as a miss measures your labelling, not your detector.

THE TAUTOLOGY CHECK

Week 5 and Week 6 spent two readings on a metric whose value could not fall:
`citation_precision` computed only over citations the system had itself
produced, which is always 1.0 and looks like success. The rule that came out of
it: **read the scoring function, and for each metric ask what input would make
it drop.** If the answer is "nothing", the metric is decoration.

For the two metrics here:

    precision drops when the detector flags something the labels do not mark.
                The `negatives` cases exist to make that possible: text with
                no personal data in it at all. Without them, precision can only
                be measured on text known to contain PII, which biases it up.
    recall     drops when a labelled span is missed. The `hard` cases exist to
                make that possible: a phone number with no country code, an MRN
                at the start of a sentence, a name with a diacritic.

A labelled set made only of easy positives cannot move either number, and a
scorecard built on one is the same bug wearing a new label.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from app.presidio_layer import ENTITIES, _get_engines, _dedupe


@dataclass(frozen=True)
class LabelledSpan:
    start: int
    end: int
    entity_type: str


@dataclass(frozen=True)
class LabelledCase:
    case_id: str
    text: str
    spans: tuple[LabelledSpan, ...]
    kind: str = "positive"   # positive | negative | hard
    note: str = ""


@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_entity: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def _bump(counts: EvalCounts, entity: str, key: str) -> None:
    counts.per_entity.setdefault(entity, {"tp": 0, "fp": 0, "fn": 0})[key] += 1


def score(cases: list[LabelledCase]) -> EvalCounts:
    """Run the real analyzer over each case and count against the labels."""
    analyzer, _ = _get_engines()
    counts = EvalCounts()
    for case in cases:
        predicted = _dedupe(analyzer.analyze(text=case.text, language="en", entities=ENTITIES))
        matched_labels: set[int] = set()
        for p in predicted:
            hit = None
            for idx, lab in enumerate(case.spans):
                if idx in matched_labels:
                    continue
                if lab.entity_type == p.entity_type and _overlaps(p.start, p.end, lab.start, lab.end):
                    hit = idx
                    break
            if hit is None:
                counts.fp += 1
                _bump(counts, p.entity_type, "fp")
            else:
                matched_labels.add(hit)
                counts.tp += 1
                _bump(counts, p.entity_type, "tp")
        for idx, lab in enumerate(case.spans):
            if idx not in matched_labels:
                counts.fn += 1
                _bump(counts, lab.entity_type, "fn")
    return counts


def report(cases: list[LabelledCase]) -> dict[str, Any]:
    """The scorecard. Per-entity recall is where the useful detail is."""
    counts = score(cases)
    per_entity = {}
    for entity, c in sorted(counts.per_entity.items()):
        p_d, r_d = c["tp"] + c["fp"], c["tp"] + c["fn"]
        per_entity[entity] = {
            **c,
            "precision": round(c["tp"] / p_d, 3) if p_d else None,
            "recall": round(c["tp"] / r_d, 3) if r_d else None,
        }
    return {
        "cases": len(cases),
        "kinds": {k: sum(1 for c in cases if c.kind == k) for k in ("positive", "negative", "hard")},
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": round(counts.precision, 3),
        "recall": round(counts.recall, 3),
        "f1": round(counts.f1, 3),
        "per_entity": per_entity,
    }


def assert_set_can_fail(cases: list[LabelledCase]) -> None:
    """The tautology check, as an assertion rather than as advice.

    A labelled set with no negatives cannot lower precision, and a set with no
    hard cases is unlikely to lower recall. Either way the scorecard reports a
    number that cannot move, which is exactly the defect Weeks 5 and 6 spent
    two readings on.
    """
    if not any(c.kind == "negative" for c in cases):
        raise AssertionError(
            "no negative cases: precision cannot drop, so the number it reports is not a measurement"
        )
    if not any(c.kind == "hard" for c in cases):
        raise AssertionError(
            "no hard cases: recall is measured only on text the detector was always going to catch"
        )
    if not any(c.spans for c in cases):
        raise AssertionError("no labelled spans anywhere: recall has no denominator")


def load(path: str | pathlib.Path) -> list[LabelledCase]:
    raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return [
        LabelledCase(
            case_id=c["case_id"],
            text=c["text"],
            spans=tuple(LabelledSpan(**s) for s in c.get("spans", [])),
            kind=c.get("kind", "positive"),
            note=c.get("note", ""),
        )
        for c in raw
    ]
