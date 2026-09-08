"""The human-adjudicated slice, in miniature (guided lab, Part 2).

Two steps, run from the repository root against a saved run
(`POST /run` with `save_results_to`, or `pytest -m live`, which does not
save; use the API for this exercise).

  1. Show ten cases WITHOUT the judges' verdicts, so you can score them
     blind against the rubric in app/judges.py:

         python scripts/human_slice.py show runs/baseline.json

     Write your scores (0 to 5) into a small JSON file, one per case id:

         {"seed-transformer-001-21681e52": 5, "typo-transformer-003-a9bedc28": 4, ...}

  2. Compute agreement across all three raters, nano, mini and you, with
     Krippendorff's alpha (interval form), using `None` for the cases you
     did not score:

         python scripts/human_slice.py score runs/baseline.json my_scores.json

The sample is one case per seed, rotating through the six strategies in seed
order, so it covers every strategy and is fixed for a given run. Ten cases rehearse the method; they do not support a
conclusion. A production slice is a few dozen cases, every run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.scorecard import krippendorff_alpha_interval, load_results  # noqa: E402


from app.schemas import Strategy  # noqa: E402

# The seed case first, then every strategy in enum order. A strategy added to
# the enum (Project 14) joins the rotation without any change here.
STRATEGY_ORDER = ["seed"] + [s.value for s in Strategy if s.value != "seed"]


def pick(results, limit: int = 10):
    """One case per seed, rotating through the strategies in seed order.

    Seed 1 contributes its seed case, seed 2 its first typo case, seed 3 its
    jailbreak case, and so on, so the ten cases cover every strategy and the
    sample is fixed for a given run.
    """
    by_seed: dict[str, list] = {}
    for r in results:
        by_seed.setdefault(r.case.seed_id, []).append(r)
    out = []
    for i, (sid, cases) in enumerate(by_seed.items()):
        want = STRATEGY_ORDER[i % len(STRATEGY_ORDER)]
        chosen = next((c for c in cases if c.case.strategy.value == want), cases[0])
        out.append(chosen)
        if len(out) >= limit:
            break
    return out


def show(path: str) -> None:
    results = load_results(path)
    for i, r in enumerate(pick(results), 1):
        print(f"\n=== case {i}: {r.case.id}")
        print(f"strategy          : {r.case.strategy.value}")
        print(f"expected behaviour: {r.case.expected_behaviour.value}")
        print(f"prompt            : {r.case.prompt}")
        if r.case.injected_context:
            print(f"injected context  : {r.case.injected_context[:300]}")
        ans = r.response.answer or "<empty: the service declined>"
        print(f"answer            : {ans[:600]}")
    print("\nScore each case 0 to 5 against the rubric in app/judges.py, then save as JSON: {case_id: score}.")


def score(path: str, scores_path: str) -> None:
    results = load_results(path)
    mine = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    ratings = []
    matched = 0
    for r in results:
        judges = list(r.judge_scores)
        human = mine.get(r.case.id)
        if human is not None:
            matched += 1
        ratings.append(judges + [float(human) if human is not None else None])
    alpha_all = krippendorff_alpha_interval(ratings)
    two_judges = krippendorff_alpha_interval([list(r.judge_scores) for r in results])
    sliced = [row for row, r in zip(ratings, results) if mine.get(r.case.id) is not None]
    alpha_slice = krippendorff_alpha_interval(sliced) if sliced else float("nan")
    print(f"cases in run        : {len(results)}")
    print(f"cases you scored    : {matched}")
    print(f"alpha, two judges   : {two_judges:.3f}")
    print(f"alpha, three raters : {alpha_all:.3f}   (your rows carry a third score; the rest carry None)")
    print(f"alpha on your slice : {alpha_slice:.3f}   (three raters, your {matched} cases only)")
    print("\nA gap between the first two numbers is what the human slice exists to show.")
    print("Ten cases cannot support a conclusion. A production slice is a few dozen cases, every run.")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "show":
        show(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
