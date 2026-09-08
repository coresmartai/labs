# No scorecard on record

**This file is a placeholder. It is not a run, and it is deliberately not pretending to be one.**

`scorecards/last.md` is written by the live eval suite. Until you run it, there is
nothing here to read:

```bash
# 1. bring the SUT up (self-contained mode: BreakRAG evaluates its own /answer)
uvicorn app.main:app --reload

# 2. in a second shell, run the four floors against it
pytest -m live tests/test_eval_harness.py -v
```

That writes a real scorecard over this file: the run manifest, all three leakage
checks, per-strategy pass rates, Krippendorff's α, the judge-disagreement rate,
and the overall verdict. In CI, the nightly `eval` workflow does the same thing
and posts the result as a PR comment.

---

### Why this file is empty rather than green

It previously shipped a **RED** scorecard from a pre-fix run - `multi_hop 0.000`,
`conflict 0.100`, `hostile 0.100`, `α 0.396` - every one of them below its floor.
Those numbers were real, and they were the harness correctly reporting that
something was broken. What was broken turned out to be the **harness**, not the
system it was testing:

- the retrieval gate saw the *whole adversarial prompt*, so a distractor or an
  insult dragged the query vector off the passage that answered it, and the SUT
  refused questions it could answer perfectly well. Multi-hop and hostile were
  structurally unable to pass, no matter how good the SUT was;
- the abstention detector recognised only one exact sentence, so a system that
  correctly said *"the sources conflict"* was scored as though it had fabricated;
- and the generator handed `expected_behaviour=MATCH` to the two out-of-domain
  refusal probes, punishing the SUT for correctly declining a question it had no
  source for.

All three are fixed - see `app/query_focus.py`, `app/generator.py` and
`app/adversarial.py`. But a scorecard is a *measurement*, and the honest thing to
do with a measurement you have not taken is to leave it blank. **Run the suite.
Read what it says.**

If it comes back RED, that is the harness doing its job. Read
`leakage_check_notes` first, then the per-strategy rows, then α. A RED scorecard
you understand is worth more than a green one you were handed.
