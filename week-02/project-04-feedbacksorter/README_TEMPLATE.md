# Project 4 - FeedbackSorter

One or two lines: what this service does and who it is for.

## Overview
Reads a piece of inbound product feedback and returns two labels, an **intent**
and an **urgency**, then benchmarks at least two providers on your own
hand-labelled golden set so the model choice is settled by numbers. Built by
re-pointing IntentIQ: same provider abstraction and benchmark harness, a new
two-field schema, and a golden set you build yourself.

Label sets (use these exactly, so grading is consistent):
- intent: `bug_report`, `feature_request`, `praise`, `billing_issue`, `other`
- urgency: `low`, `medium`, `high`

## Setup
- Python 3.10+ in a virtual environment
- `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and set `OPENAI_API_KEY`
- `.env` is gitignored. Never commit secrets.

## Run
```
python -m app.main benchmark        # runs your golden set across the providers
# or: uvicorn app.main:app --reload  # http://localhost:8000
```

## The golden set (you build this)
`data/feedback_pool.jsonl` holds a raw, UNLABELLED pool of real feedback. Hand-label
**30 rows** into `golden_dataset.jsonl`, each with both fields and a one-line note:
```
{"id": 7, "input": "app double-charged me and support is ignoring me",
 "intent": "billing_issue", "urgency": "high",
 "note": "money lost + being ignored, so high; billing, not a bug"}
```
No labelled dataset is provided on purpose: building trustworthy ground truth is
the graded skill this week. Include the hard cases (slang, ambiguity, junk), and
use every intent and every urgency level.

## Your one-line urgency rubric
State what low / medium / high mean, so your labels stay consistent. For example:
high = someone is blocked or losing money now; medium = a real problem that can
wait a day; low = nice-to-have, praise, or idle comment. Adjust, but write it down.

## API / CLI
- `POST /classify` - one piece of feedback in; returns intent, urgency, confidence.
- `python -m app.main benchmark` - scores at least `gpt-5.4-mini` and `gpt-5.4-nano`
  on your golden set and prints intent accuracy, urgency accuracy, p50, p95, cost.

## The decision
A short section: name the one constraint a feedback-triage tool cannot compromise
on, pick a provider, and cite the numbers from your benchmark that justify it.

## Design decisions
- Schema: the two-field Result, and how urgency is validated.
- The re-point: what changed from IntentIQ (schema, labels, prompt, scorer, printer)
  and what you left alone (the adapters, the harness, the dated pins).
- Failure handling: bad model output, a value outside the label set, an empty input.

## How I verified it
Which providers you ran, and how you checked the golden set and the two accuracies.

## Known limitations
Be honest. What does it not handle well yet?

## AI assistance
Which AI tools you used and how (this is expected and fine; say what you did).
