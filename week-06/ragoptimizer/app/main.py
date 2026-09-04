"""FastAPI service entry point.

Routes are the same shape as the W5 CitationRAG service. Only the retrieval
path behind /ask has changed. The citation contract (every cited chunk_id
must exist in the chunks we passed in) is preserved end-to-end.

New in w06v02c01 vs the original:
  GET /        - serves index.html browser UI
  GET /readme  - renders README.md as dark-themed HTML
  GET /health  - extended to include 'model' and 'models' keys for the health chip
  POST /eval         - runs golden_dataset.json through the full pipeline, returns metrics
  POST /eval/compare - runs BOTH baseline (W5 hybrid) and full W6 pipeline side-by-side
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.citation import ask_with_citations
from app.config import get_settings
from app.llm import REFUSAL_STRING
from app.retriever import retrieve_baseline_gated, retrieve_with_trace
from app.schemas import (
    CompareMetrics,
    CompareRow,
    EvalMetrics,
    EvalRow,
    GoldenRow,
    GroundedAnswer,
    PipelineMetrics,
    PipelineTrace,
    QueryRequest,
)

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("ragoptimizer")

app = FastAPI(title="RAGOptimizer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_HERE = Path(__file__).parent   # app/
_ROOT = _HERE.parent            # w06v02c01/


# ── Static UI ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    """Serve the browser demo UI."""
    return FileResponse(_ROOT / "index.html")


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    """Render README.md as a dark-themed HTML page."""
    import markdown as _md  # lazy import - only needed when this route is hit
    readme = _ROOT / "README.md"
    if not readme.exists():
        raise HTTPException(status_code=404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    return Response(content=_readme_html(body), media_type="text/html; charset=utf-8")


def _readme_html(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>RAGOptimizer - README</title>
<style>
  body {{ background:#0d1117; color:#e6edf3;
         font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         font-size:15px; line-height:1.75; max-width:860px; margin:0 auto; padding:40px 24px; }}
  h1,h2,h3 {{ color:#e6edf3; border-bottom:1px solid #30363d; padding-bottom:6px; margin-top:32px; }}
  h1 {{ font-size:26px; }} h2 {{ font-size:20px; }} h3 {{ font-size:16px; border:none; }}
  a {{ color:#58a6ff; }}
  code {{ background:#161b22; padding:2px 6px; border-radius:4px;
          font-family:'Fira Code',Consolas,monospace; font-size:13px; }}
  pre {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
         padding:16px; overflow-x:auto; }}
  pre code {{ background:none; padding:0; }}
  table {{ border-collapse:collapse; width:100%; margin:16px 0; }}
  th,td {{ border:1px solid #30363d; padding:8px 14px; text-align:left; }}
  th {{ background:#161b22; font-weight:600; }}
  blockquote {{ border-left:3px solid #58a6ff; margin:12px 0; padding:4px 16px; color:#7d8590; }}
  hr {{ border:none; border-top:1px solid #30363d; margin:24px 0; }}
</style>
</head>
<body>{body}</body>
</html>"""


# ── Threshold gate ─────────────────────────────────────────────────────────────

def _gate_channels(trace: PipelineTrace) -> tuple[bool, bool]:
    """Apply Week 5's threshold gate to each of the full pipeline's two channels.

    Returns (raw_ok, hyde_ok). The pipeline answers iff `raw_ok or hyde_ok`:

        raw_ok  = top1_raw  >= similarity_threshold and spread_raw  >= spread_delta
        hyde_ok = top1_hyde >= similarity_threshold and spread_hyde >= spread_delta

    Same rule, same thresholds the W5 baseline uses - the only difference is that
    W6 gets to present two probes to it. That asymmetry of *inputs* (not of rules)
    is the whole design:

      - Gating W6 on the raw query alone would reproduce the baseline's decision
        on every row by construction - same input, same thresholds - and test
        nothing.
      - Giving W6 both channels is fair: HyDE is a genuine second chance at
        clearing the gate, and a second probe is what Week 6 actually built.
      - It is not free. A drifted HyDE probe can clear the gate on the wrong
        chunk and convert a safe refusal into a confident wrong answer
        (false_answer). That trade is the week's teaching point.

    A disabled or failed HyDE probe leaves the hyde pair at (0.0, 0.0) → hyde_ok
    is False. Failing open would make the gate decorative.
    """
    s = get_settings()
    raw_ok = (
        trace.top1_raw >= s.similarity_threshold
        and trace.spread_raw >= s.spread_delta
    )
    hyde_ok = (
        trace.top1_hyde >= s.similarity_threshold
        and trace.spread_hyde >= s.spread_delta
    )
    return raw_ok, hyde_ok


def _log_gate(label: str, trace: PipelineTrace, raw_ok: bool, hyde_ok: bool) -> None:
    """Log the gate decision with all four floats and the channel that cleared."""
    s = get_settings()
    if raw_ok and hyde_ok:
        cleared = "raw+hyde"
    elif raw_ok:
        cleared = "raw"
    elif hyde_ok:
        cleared = "hyde"
    else:
        cleared = "none"
    logger.info(
        "%s gate: raw(top1=%.3f spread=%.3f)=%s hyde(top1=%.3f spread=%.3f)=%s "
        "→ cleared=%s thresholds(sim>=%.2f spread>=%.2f)",
        label,
        trace.top1_raw, trace.spread_raw, raw_ok,
        trace.top1_hyde, trace.spread_hyde, hyde_ok,
        cleared, s.similarity_threshold, s.spread_delta,
    )


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe - includes model keys for the UI health chip."""
    s = get_settings()
    return {
        "status": "ok",
        "service": "ragoptimizer",
        "model": s.model_id,
        "models": {"main": s.model_id, "hyde": s.hyde_model},
    }


@app.get("/config")
def config_snapshot() -> dict[str, object]:
    """Surface the active configuration - useful for confirming what's wired."""
    s = get_settings()
    return {
        "model_id": s.model_id,
        "hyde_model": s.hyde_model,
        "hyde_enabled": s.hyde_enabled,
        "hyde_prompt_voice": s.hyde_prompt_voice,
        "reranker_enabled": s.reranker_enabled,
        "reranker_model": s.reranker_model,
        "reranker_batch_size": s.reranker_batch_size,
        "compressor_enabled": s.compressor_enabled,
        "compressor_keep_fraction": s.compressor_keep_fraction,
        "vector_top_k_wide": s.vector_top_k_wide,
        "vector_top_k_narrow": s.vector_top_k_narrow,
        # Threshold gate - read by BOTH /eval/compare columns and by /ask.
        # The baseline gates on the raw query; the full pipeline gates on
        # (raw_ok or hyde_ok) at these same values.
        "similarity_threshold": s.similarity_threshold,
        "spread_delta": s.spread_delta,
    }


@app.post("/ask", response_model=GroundedAnswer)
def ask(req: QueryRequest) -> GroundedAnswer:
    """Retrieve relevant chunks then generate a citation-grounded answer.

    Pipeline:
        1. retrieve_with_trace() - HyDE + hybrid dense/BM25 + rerank + compress
        2. threshold gate        - refuse before the LLM unless (raw_ok or hyde_ok);
                                   Week 5's rule and values (similarity_threshold
                                   0.55, spread_delta 0.08) applied to BOTH of the
                                   pipeline's probes (see _gate_channels)
        3. ask_with_citations()  - citation prompt → gpt-5.4-mini → validate

    The gate is here, not only in /eval/compare, so the endpoint and the eval
    cannot disagree about what the pipeline does. A refusal returns the normal
    GroundedAnswer shape with fallback_triggered=True and the trace attached, so
    the UI can render WHY it refused - all four gate floats are on the trace.
    """
    chunks, trace = retrieve_with_trace(req.query)

    raw_ok, hyde_ok = _gate_channels(trace)
    _log_gate("/ask", trace, raw_ok, hyde_ok)
    if not (raw_ok or hyde_ok):
        return GroundedAnswer(
            answer=REFUSAL_STRING,
            citations=[],
            confidence="low",
            fallback_triggered=True,
            pipeline_trace=trace,
        )

    if not chunks:
        raise HTTPException(status_code=503, detail="retrieval returned no candidates")
    return ask_with_citations(req.query, chunks, trace=trace)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _judge_row(row: GoldenRow, resp: GroundedAnswer) -> dict[str, "bool | None"]:
    """Score one golden-dataset row against the pipeline response.

    Returns: grounded, cit_valid_ok, cit_recall_ok, false_answer, false_refusal.
    Every value is a bool except cit_valid_ok, which is bool | None.

    Convention (inherited from Week 5):
      must_cite == []          → question should be refused (out-of-domain)
      must_cite == ["_answered"] → any non-empty citation list is sufficient

    W6 fallback_triggered semantics:
      fallback_triggered=True means "the LLM could not fully answer from the chunks".
      This is the correct signal to use as "refused" - it fires for both out-of-domain
      questions (no relevant content) AND partial answers where the LLM declares it
      cannot answer.  The compressor_keep_fraction=0.80 prevents false positives on
      answerable questions by ensuring critical sentences are preserved before the LLM
      call, so the LLM rarely triggers fallback on questions the corpus can answer.

      Note: the LLM sometimes cites chunks even while refusing (it cites what it was
      given). Do NOT use len(citations) > 0 as a signal for "answered" - it is not
      reliable for out-of-domain questions.

    cit_valid_ok is CITATION VALIDITY, and it is tri-state.

    The first draft of this file scored `citation_precision` as

        cited_ids.issubset(must_cite_set | cited_ids)

    which is True for every possible input: a set is always a subset of a union
    containing itself. That is the same judge-that-cannot-say-no this course
    caught in Week 5, and it came back in the very next build. Week 5's rule is
    the fix and the habit: for every metric you report, ask what input would
    make this number drop. If nothing can, it is a decoration.

    What CAN fail is the model's citation discipline, measured UPSTREAM of the
    enforcement that hides it. The validator drops any citation naming a chunk
    the model was never sent; GroundedAnswer.citations_dropped counts them.

        True  - the model generated an answer and every citation it emitted
                named a chunk it was actually given
        False - the model generated an answer and fabricated at least one
                chunk reference, which the validator then removed
        None  - the model never generated (threshold gate, or a failed model
                call), so citation discipline was never exercised

    Rows scoring None leave the citation_validity denominator entirely. A row
    the gate refused is not evidence of good citation behaviour, and counting
    it as a pass is how a denominator lies.

    grounded remains a token-overlap proxy, and correct refusals count as
    grounded.
    """
    expected_refusal = len(row.must_cite) == 0
    actually_refused = resp.fallback_triggered

    # Citation validity, measured upstream of the validator that hides it.
    cit_valid: bool | None
    if not resp.generated:
        cit_valid = None                       # never exercised
    else:
        cit_valid = resp.citations_dropped == 0

    if expected_refusal and actually_refused:
        return {"grounded": True, "cit_valid_ok": cit_valid, "cit_recall_ok": True,
                "false_answer": False, "false_refusal": False}

    if expected_refusal and not actually_refused:
        return {"grounded": False, "cit_valid_ok": cit_valid, "cit_recall_ok": False,
                "false_answer": True, "false_refusal": False}

    if not expected_refusal and actually_refused:
        return {"grounded": False, "cit_valid_ok": cit_valid, "cit_recall_ok": False,
                "false_answer": False, "false_refusal": True}

    # Both expected answer and got answer - score quality
    grounded = any(
        token in resp.answer.lower()
        for token in row.expected_answer.lower().split()
        if len(token) > 4
    )
    cited_ids = {c.chunk_id for c in resp.citations}
    must_cite_set = set(row.must_cite)

    if must_cite_set == {"_answered"}:
        # Sentinel: chunk IDs are renumbered doc#1..k per query, so a specific
        # ID cannot be hardcoded. Any citation counts as a recall hit.
        cit_recall_ok = len(resp.citations) > 0
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
    """Run every question in golden_dataset.json through the full RAGOptimizer
    pipeline - threshold gate included, exactly as /ask serves it - and return a
    five-metric dashboard plus per-row detail with full pipeline traces.

    The gate refuses before the LLM unless (raw_ok or hyde_ok); see
    _gate_channels for the rule and why it reads both channels.

    Honest framing: these are deterministic proxy metrics, not statistics.
    Every rate uses ALL rows as the denominator - with 10 rows, one row moves
    any metric by exactly 0.10.

    Metrics (all in [0, 1]). Note the denominators are NOT all the same, and
    that is deliberate: each rate divides by the rows that actually exercised
    the behaviour it measures.
      groundedness        - token-overlap proxy vs the expected answer;
                            correct refusals count as grounded. Over all rows
      citation_validity   - of the rows where the model GENERATED an answer,
                            how often did every citation name a chunk it was
                            actually sent? Gate refusals never generated, so
                            they leave this denominator. Over generated rows
      citation_recall     - rows where required IDs were actually cited
      false_answer_rate   - should-refuse rows that got an answer. Over all rows
      false_refusal_rate  - should-answer rows that were refused. Over all rows
    """
    s = get_settings()
    path = Path(s.golden_dataset_path)
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Golden dataset not found: {path}. "
                   f"Expected at {path.resolve()}",
        )

    rows_raw = json.loads(path.read_text(encoding="utf-8"))
    rows = [GoldenRow(**r) for r in rows_raw]

    tallies: dict[str, int] = {
        "grounded": 0, "cit_recall_ok": 0,
        "false_answer": 0, "false_refusal": 0,
    }
    # citation_validity divides only by the rows that exercised it. Rows the
    # gate refused never generated, so they are not evidence either way.
    cit_valid_true = 0
    cit_valid_total = 0
    n = len(rows)
    eval_rows: list[EvalRow] = []

    for i, row in enumerate(rows):
        logger.info("Eval row %d/%d: %.60s", i + 1, n, row.question)
        try:
            chunks, trace = retrieve_with_trace(row.question)
            # Same gate as /ask and as /eval/compare's full column. /eval scores
            # the full pipeline, so it must see the pipeline /ask actually serves
            # - an eval that skips the gate measures a system nobody runs.
            raw_ok, hyde_ok = _gate_channels(trace)
            _log_gate(f"Eval row {i + 1}", trace, raw_ok, hyde_ok)

            if not (raw_ok or hyde_ok):
                resp = GroundedAnswer(
                    answer=REFUSAL_STRING,
                    citations=[],
                    confidence="low",
                    fallback_triggered=True,
                    pipeline_trace=trace,
                )
            elif not chunks:
                resp = GroundedAnswer(
                    answer="No relevant chunks found for this question.",
                    citations=[],
                    confidence="low",
                    fallback_triggered=True,
                    pipeline_trace=trace,
                )
            else:
                resp = ask_with_citations(row.question, chunks, trace=trace)
        except Exception as exc:
            logger.error("Eval row %d failed: %s", i + 1, exc)
            resp = GroundedAnswer(
                answer="Pipeline error during evaluation.",
                citations=[],
                confidence="low",
                fallback_triggered=True,
            )

        scores = _judge_row(row, resp)
        for k in tallies:
            if scores[k]:
                tallies[k] += 1
        if scores["cit_valid_ok"] is not None:
            cit_valid_total += 1
            if scores["cit_valid_ok"]:
                cit_valid_true += 1

        expected_refusal = len(row.must_cite) == 0
        row_passed = not scores["false_answer"] and not scores["false_refusal"]
        eval_rows.append(EvalRow(
            question=row.question,
            expected="refuse" if expected_refusal else "answer",
            actual="refused" if resp.fallback_triggered else "answered",
            passed=row_passed,
            grounded=scores["grounded"],
            false_answer=scores["false_answer"],
            false_refusal=scores["false_refusal"],
            cit_recall_ok=scores["cit_recall_ok"],
            cit_valid_ok=scores["cit_valid_ok"],
            system_answer=resp.answer[:250],
            cited_ids=[c.chunk_id for c in resp.citations],
            confidence=resp.confidence,
            fallback_triggered=resp.fallback_triggered,
            pipeline_trace=resp.pipeline_trace,
        ))

    safe = max(n, 1)
    return EvalMetrics(
        groundedness=tallies["grounded"] / safe,
        # Vacuously 1.0 when nothing generated: the behaviour was never
        # exercised, so it cannot have failed. rows[] makes that visible.
        citation_validity=(cit_valid_true / cit_valid_total if cit_valid_total else 1.0),
        citation_recall=tallies["cit_recall_ok"] / safe,
        false_answer_rate=tallies["false_answer"] / safe,
        false_refusal_rate=tallies["false_refusal"] / safe,
        rows_scored=n,
        rows=eval_rows,
    )


@app.post("/eval/compare", response_model=CompareMetrics)
def eval_compare() -> CompareMetrics:
    """Run every golden question through BOTH pipelines, sequentially row by
    row, and return paired results.

    Baseline - Week 5 CitationRAG, retrieval AND refusal:
        embed query → Qdrant dense ANN
        scroll all  → BM25 in-memory
        RRF fusion  → top-k by rank position
        threshold gate → refuse when top1_dense < similarity_threshold (0.55)
                         or spread < spread_delta (0.08), before the LLM is called
        (no HyDE, no cross-encoder rerank, no extractive compression)
        The gate is W5's, values and all, and it is what makes this a real
        'before' column: without it the baseline cannot refuse, so the
        false-refusal metric measures nothing and every column saturates.

    Full - Week 6 RAGOptimizer:
        HyDE probe  → dense ANN (semantic probe)
        hybrid      → dense+BM25+RRF (same as baseline but augmented)
        union+dedupe → cross-encoder rerank → extractive compression
        threshold gate → the SAME rule at the SAME values as the baseline, but
                         evaluated against BOTH probes:
                             raw_ok  = top1_raw  >= 0.55 and spread_raw  >= 0.08
                             hyde_ok = top1_hyde >= 0.55 and spread_hyde >= 0.08
                             refuse unless (raw_ok or hyde_ok)
                         Refuses before the LLM is called, same as the baseline.

    Why the gate is symmetric
    -------------------------
    Both columns gate, so the delta measures retrieval quality rather than gate
    removal. When only the baseline gated, its refusal rows were lost to the
    full pipeline partly because the full pipeline could not refuse at all - the
    'after' column won by having no gate, which is not a Week 6 achievement.

    The channels are deliberately asymmetric while the RULE is not. Gating W6 on
    the raw query alone would reproduce the baseline's decision on every row by
    construction (same input, same thresholds) and measure nothing; giving W6
    both channels is fair, because a second probe is what Week 6 built. HyDE
    earns its place exactly on the rows where the raw query fails the gate and
    the probe clears it - and pays for it when a drifted probe clears the gate on
    the wrong chunk, turning a safe refusal into a confident wrong answer
    (false_answer). See _gate_channels.

    The delta between baseline and full metrics isolates the value each
    Week 6 component adds.  This is the core teaching point of the module.
    Metrics are the same deterministic proxies as /eval (see eval_endpoint);
    no bootstrap statistics or confidence intervals are computed here.
    """
    s = get_settings()
    path = Path(s.golden_dataset_path)
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Golden dataset not found: {path}. "
                   f"Expected at {path.resolve()}",
        )

    rows_raw = json.loads(path.read_text(encoding="utf-8"))
    rows = [GoldenRow(**r) for r in rows_raw]
    n = len(rows)

    _zero = {"grounded": 0, "cit_recall_ok": 0,
             "false_answer": 0, "false_refusal": 0}
    b_tallies: dict[str, int] = dict(_zero)
    f_tallies: dict[str, int] = dict(_zero)
    # Separate citation-validity denominators: the two pipelines refuse
    # different rows, so they exercise citation discipline different numbers
    # of times. Sharing one denominator would compare unlike things.
    b_valid = {"true": 0, "total": 0}
    f_valid = {"true": 0, "total": 0}
    compare_rows: list[CompareRow] = []

    for i, row in enumerate(rows):
        logger.info("Compare row %d/%d: %.60s", i + 1, n, row.question)

        # ── Baseline: W5 hybrid + W5 threshold gate ────────────────────────────
        try:
            b_chunks, b_top1, b_spread = retrieve_baseline_gated(row.question)
            logger.info(
                "Baseline row %d: %d chunks top1=%.3f spread=%.3f",
                i + 1, len(b_chunks), b_top1, b_spread,
            )
            # W5's gate, verbatim: refuse before the model is called.
            # fallback_triggered=True is what _judge_row counts as "refused".
            if (
                not b_chunks
                or b_top1 < s.similarity_threshold
                or b_spread < s.spread_delta
            ):
                logger.info(
                    "Baseline row %d refusal: threshold_gate "
                    "(top1=%.3f < %.2f or spread=%.3f < %.2f)",
                    i + 1, b_top1, s.similarity_threshold, b_spread, s.spread_delta,
                )
                b_resp = GroundedAnswer(
                    answer=REFUSAL_STRING,
                    citations=[], confidence="low", fallback_triggered=True,
                )
            else:
                b_resp = ask_with_citations(row.question, b_chunks)
        except Exception as exc:
            logger.error("Baseline row %d failed: %s", i + 1, exc)
            b_resp = GroundedAnswer(
                answer="Baseline pipeline error.", citations=[],
                confidence="low", fallback_triggered=True,
            )

        # ── Full W6 pipeline: same gate, two channels ──────────────────────────
        try:
            f_chunks, f_trace = retrieve_with_trace(row.question)
            raw_ok, hyde_ok = _gate_channels(f_trace)
            _log_gate(f"Full row {i + 1}", f_trace, raw_ok, hyde_ok)

            if not (raw_ok or hyde_ok):
                # Same rule, same thresholds, same refusal shape as the baseline
                # branch above - refuse BEFORE the LLM is called. The only thing
                # W6 gets that the baseline does not is a second probe to clear
                # the gate with.
                f_resp = GroundedAnswer(
                    answer=REFUSAL_STRING,
                    citations=[], confidence="low", fallback_triggered=True,
                    pipeline_trace=f_trace,
                )
            elif not f_chunks:
                f_resp = GroundedAnswer(
                    answer="No candidates found for this question.",
                    citations=[], confidence="low", fallback_triggered=True,
                    pipeline_trace=f_trace,
                )
            else:
                f_resp = ask_with_citations(row.question, f_chunks, trace=f_trace)
        except Exception as exc:
            logger.error("Full pipeline row %d failed: %s", i + 1, exc)
            f_resp = GroundedAnswer(
                answer="Full pipeline error.", citations=[],
                confidence="low", fallback_triggered=True,
            )

        b_scores = _judge_row(row, b_resp)
        f_scores = _judge_row(row, f_resp)

        for metric in b_tallies:
            if b_scores[metric]:
                b_tallies[metric] += 1
        for metric in f_tallies:
            if f_scores[metric]:
                f_tallies[metric] += 1
        for scores, acc in ((b_scores, b_valid), (f_scores, f_valid)):
            if scores["cit_valid_ok"] is not None:
                acc["total"] += 1
                if scores["cit_valid_ok"]:
                    acc["true"] += 1

        expected_refusal = len(row.must_cite) == 0
        compare_rows.append(CompareRow(
            question=row.question,
            expected="refuse" if expected_refusal else "answer",
            baseline_actual="refused" if b_resp.fallback_triggered else "answered",
            baseline_passed=not b_scores["false_answer"] and not b_scores["false_refusal"],
            baseline_answer=b_resp.answer[:250],
            baseline_cited_ids=[c.chunk_id for c in b_resp.citations],
            baseline_confidence=b_resp.confidence,
            baseline_fallback=b_resp.fallback_triggered,
            full_actual="refused" if f_resp.fallback_triggered else "answered",
            full_passed=not f_scores["false_answer"] and not f_scores["false_refusal"],
            full_answer=f_resp.answer[:250],
            full_cited_ids=[c.chunk_id for c in f_resp.citations],
            full_confidence=f_resp.confidence,
            full_fallback=f_resp.fallback_triggered,
            full_pipeline_trace=f_resp.pipeline_trace,
        ))

    safe = max(n, 1)

    def _metrics(t: dict[str, int], v: dict[str, int]) -> PipelineMetrics:
        return PipelineMetrics(
            groundedness=t["grounded"] / safe,
            citation_validity=(v["true"] / v["total"] if v["total"] else 1.0),
            citation_recall=t["cit_recall_ok"] / safe,
            false_answer_rate=t["false_answer"] / safe,
            false_refusal_rate=t["false_refusal"] / safe,
        )

    return CompareMetrics(
        rows_scored=n,
        baseline=_metrics(b_tallies, b_valid),
        full=_metrics(f_tallies, f_valid),
        rows=compare_rows,
    )
