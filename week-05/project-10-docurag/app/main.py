"""FastAPI routes for CitationRAG.

Two endpoints:
  POST /answer  - answer a question with citations or refuse
  POST /eval    - run the generated golden dataset (scripts/build_golden_dataset.py)
                  through /answer and report the five-metric dashboard

The handler never touches the SDK directly - every model call goes through
app/llm.py. Replacing the LLM provider is a one-file change.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.citations import validate
from app.config import get_settings
from app.llm import REFUSAL_STRING, generate_with_citations
from app.retriever import Retriever
from app.schemas import (
    AnswerRequest,
    AnswerResponse,
    EvalMetrics,
    EvalRow,
    GoldenRow,
)

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("citationrag")

app = FastAPI(title="CitationRAG", version="0.1.0")

# CORS open for local development - narrow allow_origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    """Return the shared Retriever instance (connects to Qdrant on first call)."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _refuse(
    top: float | None = None,
    spread: float | None = None,
    source: str | None = None,
    chunks: list | None = None,
    raw_answer: str | None = None,
    raw_citations: list | None = None,
    detail: str | None = None,
) -> AnswerResponse:
    """Build a refused AnswerResponse with the exact refusal string.

    Centralises refusal construction so every call site gets identical fields.
    `source` identifies which stage fired: "threshold_gate", "llm", or "validation".
    `raw_answer` / `raw_citations` / `detail` are only set for validation failures -
    they preserve the model's original output so the UI can show what broke.
    """
    return AnswerResponse(
        answer=REFUSAL_STRING,
        citations=[],
        refused=True,
        retrieval_top_score=top,
        retrieval_spread=spread,
        validation_passed=True,
        refusal_source=source,
        retrieved_chunks=chunks,
        raw_model_answer=raw_answer,
        raw_model_citations=raw_citations,
        validation_detail=detail,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    """Retrieve → threshold gate → generate-with-citations → validate."""
    settings = get_settings()
    retriever = _get_retriever()

    chunks, top1, spread = retriever.search(req.question)

    logger.info(
        "retrieval q=%r top1=%.3f spread=%.3f k=%d",
        req.question[:60], top1, spread, len(chunks),
    )

    # Threshold gate - refuse before the model is even called
    if top1 < settings.similarity_threshold or spread < settings.spread_delta:
        logger.info("refusal: below threshold or spread too narrow")
        return _refuse(top=top1, spread=spread,
                       source="threshold_gate", chunks=chunks)

    # Generate - model receives the retrieved chunks
    resp = generate_with_citations(req.question, chunks)
    resp.retrieval_top_score = top1
    resp.retrieval_spread = spread
    resp.retrieved_chunks = chunks   # always expose what the model received

    # Validate the model's citations against the actual chunks
    resp = validate(resp, chunks)

    if not resp.validation_passed:
        logger.warning(
            "validation failed - %s", resp.validation_detail or "unknown reason"
        )
        # Conservative path: convert to refusal rather than ship a bad answer.
        # Preserve the raw model output so the UI can show what actually broke.
        return _refuse(
            top=top1, spread=spread,
            source="validation", chunks=chunks,
            raw_answer=resp.answer,
            raw_citations=resp.citations,
            detail=resp.validation_detail,
        )

    if resp.refused:
        # LLM looked at the chunks and decided they didn't cover the question
        resp.refusal_source = "llm"

    logger.info(
        "answer ok citations=%d validation_passed=%s",
        len(resp.citations), resp.validation_passed,
    )
    return resp


def _judge_row(row: GoldenRow, response: AnswerResponse) -> dict[str, "bool | None"]:
    """Score one golden-dataset row against the response.

    Returns: grounded, cit_valid_ok, cit_recall_ok, false_answer, false_refusal.

    cit_valid_ok is CITATION VALIDITY, not precision-against-a-list. The
    deterministic validator already guarantees no invalid citation ever ships
    (a failure becomes a refusal with refusal_source == "validation"), so a
    "precision" check here would be True by construction - a judge that cannot
    say no is not a judge. What we CAN measure is the model's raw citation
    discipline: of the rows where the model generated an answer at all, how
    often did its citations survive validation?
        True  - model generated and every citation survived the validator
        False - model generated but the validator rejected it
        None  - model never generated (threshold gate, or a genuine LLM
                refusal) so citation discipline was not exercised
    Rows with None are excluded from the citation_validity denominator.
    """
    expected_refusal = len(row.must_cite) == 0
    actually_refused = response.refused

    # Did the model generate an answer, and did its citations survive?
    # refusal_source == "validation" means: generated, but the validator said no.
    if response.refusal_source == "validation":
        cit_valid: bool | None = False
    elif not actually_refused:
        cit_valid = True          # answered = generated + survived validation
    else:
        cit_valid = None          # threshold gate / LLM refusal: never generated

    # Refusal logic - patched after Failure #3 (see README §7)
    if expected_refusal and actually_refused:
        return {
            "grounded": True,
            "cit_valid_ok": cit_valid,
            "cit_recall_ok": True,
            "false_answer": False,
            "false_refusal": False,
        }
    if expected_refusal and not actually_refused:
        return {
            "grounded": False,
            "cit_valid_ok": cit_valid,
            "cit_recall_ok": False,
            "false_answer": True,
            "false_refusal": False,
        }
    if (not expected_refusal) and actually_refused:
        return {
            "grounded": False,
            "cit_valid_ok": cit_valid,
            "cit_recall_ok": False,
            "false_answer": False,
            "false_refusal": True,
        }

    # Substring overlap as a cheap groundedness proxy
    expected_lower = row.expected_answer.lower()
    answer_lower = response.answer.lower()
    grounded = any(
        token in answer_lower for token in expected_lower.split() if len(token) > 4
    )

    cited_ids = {c.chunk_id for c in response.citations}
    must_cite_set = set(row.must_cite)

    # "_answered" is the sentinel used when chunk IDs are dynamic (Qdrant RRF
    # renumbers them doc#1…k per query, so we can't hardcode specific IDs).
    # In that case check only that at least one citation was emitted.
    if must_cite_set == {"_answered"}:
        cit_recall_ok = len(response.citations) > 0
    else:
        cit_recall_ok = must_cite_set.issubset(cited_ids) if must_cite_set else True

    return {
        "grounded": grounded,
        "cit_valid_ok": cit_valid,
        "cit_recall_ok": cit_recall_ok,
        "false_answer": not grounded,
        "false_refusal": False,
    }


@app.post("/eval", response_model=EvalMetrics)
def eval_endpoint() -> EvalMetrics:
    """Run the generated golden dataset (scripts/build_golden_dataset.py) against /answer and compute the dashboard."""
    settings = get_settings()
    path = Path(settings.golden_dataset_path)
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Golden dataset missing: {path}. Run: python scripts/build_golden_dataset.py",
        )

    rows_raw = json.loads(path.read_text(encoding="utf-8"))
    rows = [GoldenRow(**r) for r in rows_raw]

    tallies = {
        "false_answer": 0,
        "false_refusal": 0,
    }
    # DENOMINATOR HONESTY - each rate divides only by the rows that actually
    # exercised the behaviour it measures (see README, Failure #4):
    #   groundedness      : rows that SHIPPED AN ANSWER (any answered row)
    #   citation_validity : rows where the model GENERATED (validation
    #                       refusals count against it; gate/llm refusals out)
    #   citation_recall   : rows that SHOULD cite (expected-answer rows)
    #   false_answer_rate, false_refusal_rate : all rows
    grounded_true = 0
    answered_total = 0
    cit_valid_true = 0
    cit_valid_total = 0
    recall_true = 0
    recall_total = 0
    n = len(rows)

    eval_rows: list[EvalRow] = []

    for row in rows:
        resp = answer(AnswerRequest(question=row.question))
        scores = _judge_row(row, resp)
        for k in tallies:
            if scores[k]:
                tallies[k] += 1
        if not resp.refused:
            answered_total += 1
            if scores["grounded"]:
                grounded_true += 1
        if scores["cit_valid_ok"] is not None:
            cit_valid_total += 1
            if scores["cit_valid_ok"]:
                cit_valid_true += 1
        expected_refusal = len(row.must_cite) == 0
        if not expected_refusal:
            recall_total += 1
            if scores["cit_recall_ok"]:
                recall_true += 1
        row_passed = not scores["false_answer"] and not scores["false_refusal"]
        eval_rows.append(EvalRow(
            question=row.question,
            expected="refuse" if expected_refusal else "answer",
            actual="refused" if resp.refused else "answered",
            passed=row_passed,
            grounded=scores["grounded"],
            false_answer=scores["false_answer"],
            false_refusal=scores["false_refusal"],
            cit_recall_ok=scores["cit_recall_ok"],
            cit_valid_ok=scores["cit_valid_ok"],
            system_answer=resp.answer[:250],
            cited_ids=[c.chunk_id for c in resp.citations],
            refusal_source=resp.refusal_source,
            response=resp,   # full response for expanded pipeline view
        ))

    safe = max(n, 1)
    # Empty denominators are vacuously 1.0: the behaviour was never exercised,
    # so it cannot have failed. The per-row breakdown makes this visible.
    return EvalMetrics(
        groundedness=(grounded_true / answered_total if answered_total else 1.0),
        citation_validity=(
            cit_valid_true / cit_valid_total if cit_valid_total else 1.0
        ),
        citation_recall=(recall_true / recall_total if recall_total else 1.0),
        false_answer_rate=tallies["false_answer"] / safe,
        false_refusal_rate=tallies["false_refusal"] / safe,
        rows_scored=n,
        rows=eval_rows,
    )


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the browser UI."""
    idx = Path(__file__).parent.parent / "index.html"
    if not idx.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(idx.read_text(encoding="utf-8"))


_README_CSS = (
    "body{background:#0d1117;color:#e6edf3;"
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
    "max-width:860px;margin:0 auto;padding:40px 24px;line-height:1.7}"
    "h1,h2,h3,h4{color:#e6edf3;margin-top:1.5em}"
    "h1{border-bottom:1px solid #30363d;padding-bottom:12px}"
    "h2{border-bottom:1px solid #21262d;padding-bottom:8px}"
    "a{color:#58a6ff}"
    "code{background:#21262d;padding:2px 6px;border-radius:4px;"
    "font-family:'Fira Code','Cascadia Code','Consolas',monospace;font-size:.88em}"
    "pre{background:#161b22;border:1px solid #30363d;border-radius:8px;"
    "padding:16px;overflow-x:auto;margin:1em 0}"
    "pre code{background:none;padding:0;font-size:.85em}"
    "table{border-collapse:collapse;width:100%;margin:1em 0}"
    "th,td{border:1px solid #30363d;padding:8px 12px;text-align:left}"
    "th{background:#161b22;font-weight:600}"
    "tr:nth-child(even){background:#161b22}"
    "blockquote{border-left:3px solid #30363d;margin:0 0 1em;padding:0 16px;color:#7d8590}"
    "hr{border:none;border-top:1px solid #30363d;margin:2em 0}"
    "ul,ol{padding-left:1.5em}"
    "li{margin-bottom:4px}"
)


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    """Render README.md as a dark-themed HTML page."""
    import markdown as _md

    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        raise HTTPException(status_code=404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    html = (
        "<!DOCTYPE html><html lang='en'><head>"
        "<meta charset='UTF-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>CitationRAG - README</title>"
        f"<style>{_README_CSS}</style>"
        "</head><body>"
        f"{body}"
        "</body></html>"
    )
    return Response(content=html, media_type="text/html; charset=utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "service": "citationrag", "model": settings.llm_model}
