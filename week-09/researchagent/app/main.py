"""Run ResearchAgent over your three questions and write the four artefacts.

    python -m app.main                      # uses questions.json
    python -m app.main --questions q.json   # somewhere else

Writes brief.md, ledger.json, trace.json and validation.json into the working
directory. Those four files plus your README are what you submit.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from app.agent import run_question
from app.brief import assemble_section, render_markdown
from app.ledger import LedgerProtocol, NaiveLedger
from app.retrieval import CorpusIndex
from app.schemas import Brief, ResearchResponse
from app.tools import build_tools
from app.validator import validate_brief

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("researchagent")

# RESEARCHAGENT_RFC_DIR lets a reviewer point several clones at one fetched copy
# of the corpus instead of downloading it per submission.
RFC_DIR = Path(os.environ.get(
    "RESEARCHAGENT_RFC_DIR",
    Path(__file__).resolve().parent.parent / "corpus" / "rfc"))


def build_ledger() -> LedgerProtocol:
    """THE ONE LINE TO CHANGE.

    NaiveLedger is Week 5's per-query numbering and it is wrong for a brief.
    Read app/ledger.py, write your own, and return it here instead.
    """
    return NaiveLedger()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    if len(questions) != 3:
        raise SystemExit(f"Expected three questions, found {len(questions)}. See the brief.")

    index = CorpusIndex(RFC_DIR)
    log.info("Indexed %d chunks across %d documents", len(index.chunks), len(index.documents))
    ledger = build_ledger()
    tools = build_tools(index)

    sections, all_trace, stops = [], [], []
    for qi, q in enumerate(questions):
        log.info("Question %d: %s", qi + 1, q["question"])
        trace, stop, evidence, notes = run_question(
            index, ledger, tools, q["question"], qi)
        log.info("  stopped: %s after %d iterations, %d evidence entries",
                 stop, len(trace), len(evidence))
        all_trace.extend(trace)
        stops.append(stop)
        sections.append(assemble_section(q["question"], q["answerability"], notes, evidence))

    brief = Brief(sections=sections)
    report = validate_brief(brief, ledger, index)

    out = Path(args.out)
    (out / "brief.md").write_text(render_markdown(brief))
    # brief.json is brief.md's structured twin. It exists so a reviewer can
    # re-run validation without parsing markdown, which is what makes rubric
    # line 1 checkable rather than take-your-word-for-it.
    (out / "brief.json").write_text(brief.model_dump_json(indent=2) + "\n")
    (out / "ledger.json").write_text(json.dumps(
        [e.model_dump() for e in ledger.entries()], indent=2) + "\n")
    (out / "validation.json").write_text(report.model_dump_json(indent=2) + "\n")
    response = ResearchResponse(brief=brief, ledger=ledger.entries(),
                                iter_count=len(all_trace),
                                stop_reason=",".join(stops), trace=all_trace)
    (out / "trace.json").write_text(response.model_dump_json(indent=2) + "\n")

    log.info("Wrote brief.md, brief.json, ledger.json, trace.json, validation.json")
    log.info("Validation: %s. %d/%d claims valid, %d uncited",
             "PASSED" if report.passed else "FAILED",
             report.claims_valid, report.claims_total,
             report.claims_total - report.claims_cited)
    if not report.passed:
        for v in report.verdicts:
            if not v.ok:
                log.info("  section %d claim %d: %s", v.section, v.claim, v.reason)


if __name__ == "__main__":
    main()
