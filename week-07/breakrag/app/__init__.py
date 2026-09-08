"""BreakRAG™ - adversarial eval + LLM-as-judge harness.

Built in Week 7 of the Applied GenAI & Agentic AI Engineering Course. The harness has four
moving parts in a strict pipeline:

  1. System under test  → `app/system_under_test.py`
  2. Adversarial generator → `app/adversarial.py`
  3. LLM-as-judge scorer → `app/judges.py`
  4. CI front-end + scorecard → `app/scorecard.py` + `tests/test_eval_harness.py`

Map onto the four-layer mental model:
  Model     = the judges (`app/judges.py`)
  Retrieval = the seed-set loader (`app/seeds/`)
  Tool      = the system-under-test HTTP client (`app/system_under_test.py`)
  Memory    = the scorecard history (`scorecards/`, gitignored at the repo root)
"""

__version__ = "0.1.0"
