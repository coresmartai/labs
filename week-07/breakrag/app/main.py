"""FastAPI surface for running the eval harness on demand.

  GET  /                -> serve browser UI (index.html)
  GET  /health          -> liveness probe with both judge model names
  GET  /readme          -> render README.md as dark-themed HTML
  GET  /seeds           -> the golden seed set (id, question, expected answer, behaviour)
  POST /answer          -> built-in SUT: retrieve from Qdrant + generate answer
  POST /run             -> run full suite against SUT_BASE_URL, return scorecard JSON
  GET  /run-demo-stream -> SSE demo mode: real SUT + real OpenAI judges
  POST /score-one       -> score a single question-answer pair

All three eval routes accept an optional `judge_model` override. It re-pins
the primary judge's model for that call only; without it the judge settings
from config apply.

Self-contained SUT mode
-----------------------
With a store configured (QDRANT_MODE=local and QDRANT_LOCAL_PATH, or
QDRANT_MODE=server and QDRANT_URL), BreakRAG exposes POST /answer using the
Qdrant collection (hybrid BM25 + dense retrieval, RRF fusion). Set:
    SUT_BASE_URL=http://localhost:8000
and the harness will evaluate its own /answer endpoint on every /run call.

External SUT mode
-----------------
Point SUT_BASE_URL at a running CitationRAG or RAGOptimizer
server. The harness understands both wire formats automatically.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.adversarial import expand, load_seeds
from app.config import get_settings
from app.generator import generate_answer
from app.judges import available_judges, score_with_all
from app.leakage import collect_leakage_inputs, probe_canaries
from app.query_focus import focus_queries
from app.scorecard import (
    build_scorecard,
    case_passed,
    did_expected_behaviour,
    load_results,
    save_results,
    write_markdown,
)
from app.schemas import (
    AdversarialCase,
    CaseResult,
    ExpectedBehaviour,
    Strategy,
    SUTResponse,
)
from app.system_under_test import call_sut, SystemUnderTestError

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("breakrag")

_HERE = Path(__file__).parent.parent

app = FastAPI(title="BreakRAG™", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy retriever singleton (connects to Qdrant on first /answer call)
# ---------------------------------------------------------------------------

_retriever = None


def _get_retriever():
    """Return the shared Retriever; connect to Qdrant on first call."""
    global _retriever
    if _retriever is None:
        from app.retriever import Retriever
        _retriever = Retriever()
    return _retriever


async def _scorecard_with_leakage(
    results: list[CaseResult], seeds: list, baseline_results: list[CaseResult] | None = None
) -> Any:
    """Build the scorecard with all three leakage checks fed real inputs.

    Same inputs on every path: the indexed passages are the haystack, their
    stored vectors are the embedding baseline, the seeds are embedded with the
    index's own model, and every canary-bearing seed is probed against the
    SUT. If any of that is unavailable the checks report skipped, the
    scorecard goes AMBER, and the note says why. Nothing here can turn a
    missing input into a green row.
    """
    inputs = await asyncio.to_thread(collect_leakage_inputs, seeds)
    canaries = await probe_canaries(seeds)
    if inputs.note:
        logger.info("leakage: %s", inputs.note)
    # The embedding scan is a matrix product over the corpus; keep it off the
    # event loop so /health and the SSE stream stay responsive.
    return await asyncio.to_thread(
        build_scorecard,
        results,
        baseline_results=baseline_results,
        seeds=seeds,
        leakage_haystack=inputs.haystack,
        seed_embeddings=inputs.seed_embeddings,
        corpus_embeddings=inputs.corpus_embeddings,
        canary_completions=canaries,
        leakage_inputs_note=inputs.note,
    )


# ---------------------------------------------------------------------------
# Static / meta routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the browser UI."""
    ui = _HERE / "index.html"
    if not ui.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(ui, media_type="text/html")


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe - returns both judge model names for the UI health chip."""
    s = get_settings()
    return {
        "status": "ok",
        "model": s.openai_model,
        "models": {
            "sut":   s.sut_model,
            "openai": s.openai_model,
            "nano":  s.openai_model_nano,
        },
    }


@app.get("/seeds")
def seeds_endpoint() -> list[dict[str, Any]]:
    """The golden seed set, for the browser UI's demo pills.

    Served from the one file that holds it, so the UI never carries its own
    copy of a seed question. Canaries are deliberately not included: they
    exist to be absent from everything except the seed file.
    """
    return [
        {
            "id": seed.id,
            "question": seed.question,
            "expected_answer": seed.expected_answer,
            "expected_behaviour": seed.expected_behaviour.value,
        }
        for seed in load_seeds()
    ]


class AnswerRequest(BaseModel):
    """Inbound /answer request - accepts both BreakRAG and CitationRAG shapes."""

    question: str
    adversarial_context: str | None = None   # injected by CONFLICT strategy


@app.post("/answer")
async def answer_endpoint(req: AnswerRequest) -> dict[str, Any]:
    """Built-in SUT: retrieve from Qdrant, generate a grounded answer.

    Routing only - retrieval threshold gates here; generation delegated to
    app.generator.generate_answer so main.py stays free of LLM SDK calls.

    Wire format matches what system_under_test.py expects:
      { "answer": "...", "retrieved_contexts": ["text1", "text2"],
        "refused": false, "abstained": false, "latency_ms": 150 }
    """
    s = get_settings()

    # Focus the retrieval query before searching --------------------------------
    # The user's message is not a search query. Insults, threats, urgency and
    # off-topic distractors are all just words to an embedding model, and they
    # pull the query vector away from the passage that answers it. See
    # app/query_focus.py. Falls back to the raw question if the rewrite is
    # unavailable, so this can only add recall, never remove it.
    queries = await focus_queries(req.question)

    # Retrieval - search each focused query, keep the best-grounded result -------
    try:
        retriever = _get_retriever()
        best: tuple[list, float, float] = ([], 0.0, 0.0)
        for q in queries:
            chunks_q, top1_q, spread_q = retriever.search(q)
            if top1_q > best[1]:
                best = (chunks_q, top1_q, spread_q)
        chunks, top1, _spread = best
    except Exception as exc:
        hint = (
            "Set OPENAI_API_KEY in .env and restart: the query is embedded with the "
            "index's model before the store is searched."
            if "api_key" in str(exc).lower()
            else "Check QDRANT_MODE and QDRANT_LOCAL_PATH (or QDRANT_URL) in .env, "
            "and that no other server has the embedded store open."
        )
        raise HTTPException(
            status_code=503,
            detail=f"Retriever unavailable: {exc}. {hint}",
        ) from exc

    # Two-gate check - refuse before spending tokens (mirrors CitationRAG):
    #   gate 1: top-1 dense similarity must clear SIMILARITY_THRESHOLD
    #   gate 2: top1 − top3 spread must clear SPREAD_DELTA (ambiguity guard)
    # The gate is unchanged and uncompromised: focusing the query improves recall
    # of a question the corpus CAN answer; it does not lower the bar for grounding.
    # An out-of-domain probe focuses to a clean out-of-domain query and still fails
    # gate one - which is the whole point of the refusal seeds in the golden set.
    if top1 < s.similarity_threshold or _spread < s.spread_delta or not chunks:
        logger.info(
            "answer: refusal via threshold gate (top1=%.3f spread=%.3f queries=%s)",
            top1, _spread, queries,
        )
        return {
            "answer": "",
            "retrieved_contexts": [],
            "refused": True,
            "abstained": False,
            "latency_ms": 0,
        }

    # Generation - delegated to app.generator ----------------------------------
    result = await generate_answer(
        question=req.question,
        chunks=chunks,
        adversarial_context=req.adversarial_context,
    )
    logger.info(
        "answer: q=%r top1=%.3f lat=%dms abstained=%s",
        req.question[:60], top1, result["latency_ms"], result["abstained"],
    )
    return {"refused": False, **result}


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    """Render README.md as a dark-themed HTML page."""
    try:
        import markdown as _md
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="markdown package not installed - run: pip install markdown==3.7",
        )
    readme = _HERE / "README.md"
    if not readme.exists():
        raise HTTPException(status_code=404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    page = (
        "<!DOCTYPE html><html lang=\"en\"><head>"
        "<meta charset=\"UTF-8\"><title>BreakRAG&#8482; README</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "max-width:900px;margin:40px auto;padding:0 24px;background:#0d1117;"
        "color:#e6edf3;line-height:1.7;font-size:15px}"
        "h1,h2,h3{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:6px;margin-top:32px}"
        "code{background:#21262d;padding:2px 6px;border-radius:4px;font-size:13px;"
        "font-family:'Fira Code',monospace}"
        "pre{background:#161b22;padding:16px;border-radius:8px;overflow-x:auto}"
        "pre code{background:none;padding:0}"
        "table{border-collapse:collapse;width:100%;margin:16px 0}"
        "th,td{border:1px solid #30363d;padding:8px 12px;text-align:left}"
        "th{background:#161b22;color:#58a6ff}"
        "a{color:#58a6ff}"
        "blockquote{border-left:3px solid #30363d;margin:0;padding:0 16px;color:#7d8590}"
        "</style></head><body>" + body + "</body></html>"
    )
    return Response(content=page, media_type="text/html; charset=utf-8")


# ---------------------------------------------------------------------------
# Eval routes
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    """Optional body for POST /run."""

    # Re-pin the primary judge's model for this run.
    judge_model: str | None = None
    # Save this run's per-case results here (JSON), so a later run can compare.
    save_results_to: str | None = None
    # Compare this run against a saved one: adds delta_pass_rate[strategy] rows
    # with a paired bootstrap interval. Reported, never gated.
    baseline_path: str | None = None
    # Also write the rendered scorecard (markdown) here, the same file the live
    # suite writes: scorecards/last.md is the artefact a pull request carries.
    save_scorecard_to: str | None = None


@app.post("/run")
async def run(req: RunRequest | None = None) -> dict[str, Any]:
    """Run the full red-team suite synchronously and return the scorecard."""
    judges = available_judges(judge_model=req.judge_model if req else None)
    if not judges:
        raise HTTPException(
            status_code=503,
            detail="No judges available - set OPENAI_API_KEY in .env.",
        )
    seeds = load_seeds()
    cases = expand(seeds)

    results: list[CaseResult] = []
    for case in cases:
        try:
            response = await call_sut(case)
        except SystemUnderTestError as exc:
            logger.warning("run: SUT error on case %s: %s", case.id, exc)
            continue
        verdicts = await score_with_all(judges, case, response)
        results.append(
            CaseResult(case=case, response=response, judge_verdicts=verdicts)
        )

    baseline = None
    if req and req.baseline_path:
        baseline = load_results(req.baseline_path)
    sc = await _scorecard_with_leakage(results, seeds, baseline_results=baseline)
    if req and req.save_results_to:
        save_results(results, req.save_results_to)
    if req and req.save_scorecard_to:
        Path(req.save_scorecard_to).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(sc, req.save_scorecard_to)
    return sc.model_dump(mode="json")


@app.get("/run-demo-stream")
async def run_demo_stream(
    seeds_limit: int = 0,
    cases_per_seed: int = 0,
    judge_model: str | None = None,
):
    """SSE stream: send adversarial cases to the real SUT, score with both judges.

    The SUT is whatever SUT_BASE_URL points at (.env).  Set it to
    http://localhost:8000 to evaluate BreakRAG's own /answer endpoint
    (self-contained mode with Qdrant), or to a running previous weeks server
    for external SUT mode.

    Query params (from UI controls):
      seeds_limit    int  - use first N seeds (0 = all)
      cases_per_seed int  - total adversarial cases per seed (min 6)
      judge_model    str  - optional: re-pin the primary judge's model

    Event shapes (Pattern B.1 - inline type field, no separate event: line):
      data: {"type":"start",    "n_seeds":N, "n_cases":M, "strategies":[...], "judge":"..."}
      data: {"type":"progress", "case_id":"...", "strategy":"...", "score":5.0,
                                "n":1, "total":M, "expected":"match|refuse|abstain",
                                "prompt_preview":"first 60 chars...",
                                "prompt":"full adversarial prompt",
                                "response_answer":"...", "response_refused":false,
                                "response_abstained":false, "first_context":"...",
                                "justification":"...", "passed":true}
      data: {"type":"scorecard","data":{...scorecard JSON...}}
      data: [DONE]
    """

    async def _generate():
        seeds = load_seeds()
        if 0 < seeds_limit <= len(seeds):
            seeds = seeds[:seeds_limit]
        cases = expand(
            seeds,
            cases_per_seed_override=cases_per_seed if cases_per_seed > 0 else None,
        )
        # Quick lookup: seed_id → original question (for the pipeline trace)
        seed_map = {s.id: s.question for s in seeds}

        judges = available_judges(judge_model=judge_model)
        if not judges:
            yield (
                "data: "
                + json.dumps({
                    "type":    "warn",
                    "message": "No judges available - set OPENAI_API_KEY in .env.",
                })
                + "\n\n"
            )
            yield "data: [DONE]\n\n"
            return

        s = get_settings()
        strategies = sorted({c.strategy.value for c in cases})
        judge_label = " + ".join(j.name for j in judges)
        yield (
            "data: "
            + json.dumps({
                "type":       "start",
                "n_seeds":    len(seeds),
                "n_cases":    len(cases),
                "strategies": strategies,
                "sut_url":    s.sut_base_url,
                "sut_model":  s.sut_model,
                "judge":      judge_label,
            })
            + "\n\n"
        )

        _judge_error_warned = False   # emit warn SSE at most once per run

        results: list[CaseResult] = []
        for i, case in enumerate(cases):
            # Call the real SUT - same path as /run
            try:
                response = await call_sut(case)
            except SystemUnderTestError as exc:
                logger.warning("Demo stream: SUT error on case %s: %s", case.id, exc)
                yield (
                    "data: "
                    + json.dumps({
                        "type":    "warn",
                        "message": f"SUT unreachable on case {case.id}: {exc}. "
                                   f"Check SUT_BASE_URL={s.sut_base_url} in .env.",
                    })
                    + "\n\n"
                )
                continue

            verdicts = await score_with_all(judges, case, response)

            # Detect judge API errors (e.g. billing limit) - warn once, scores stay 0
            if (
                not _judge_error_warned
                and any(v.justification.startswith("API error") for v in verdicts)
            ):
                _judge_error_warned = True
                err_detail = next(
                    v.justification for v in verdicts
                    if v.justification.startswith("API error")
                )
                logger.warning("Demo stream: judge API error. %s", err_detail)
                yield (
                    "data: "
                    + json.dumps({
                        "type":    "warn",
                        "message": (
                            f"Judge API error: {err_detail}  "
                            "Scores will show 0 for affected cases. "
                            "Check your OpenAI billing limits."
                        ),
                    })
                    + "\n\n"
                )

            result = CaseResult(case=case, response=response, judge_verdicts=verdicts)
            results.append(result)

            avg    = sum(v.score for v in verdicts) / len(verdicts) if verdicts else 0.0

            # ONE definition of "pass", shared with the scorecard (app.scorecard
            # .case_passed): the mean judge score clears 4.0 AND the SUT did the
            # expected thing. The trace and the scorecard can no longer disagree -
            # green rows above a RED scorecard was a real bug, and it was on camera.
            passed = case_passed(result)
            # Behaviour on its own is still worth showing: it separates "the system
            # did the right thing but scored badly" from "it did the wrong thing".
            behaved = did_expected_behaviour(case, response)

            delta = result.judge_delta
            needs_review = delta is not None and delta > s.judge_disagreement_delta

            yield (
                "data: "
                + json.dumps({
                    "type":               "progress",
                    "case_id":            case.id,
                    "strategy":           case.strategy.value,
                    "score":              round(avg, 1),
                    "n":                  i + 1,
                    "total":              len(cases),
                    "expected":           case.expected_behaviour.value,
                    "prompt_preview":     case.prompt[:80],
                    "prompt":             case.prompt,
                    "seed_question":      seed_map.get(case.seed_id, ""),
                    "response_answer":    response.answer[:300] if response.answer else "",
                    "response_refused":   response.refused,
                    "response_abstained": response.abstained,
                    "first_context":      (
                        response.retrieved_contexts[0]
                        if response.retrieved_contexts else ""
                    ),
                    "passed":             passed,
                    "behaved":            behaved,
                    "judge_delta":        round(delta, 1) if delta is not None else None,
                    "needs_review":       needs_review,
                    # Full per-judge breakdown for the pipeline trace
                    "verdicts": [
                        {
                            "judge":         v.judge_name,
                            "score":         v.score,
                            "justification": v.justification,
                        }
                        for v in verdicts
                    ],
                })
                + "\n\n"
            )
        yield (
            "data: "
            + json.dumps({"type": "leakage", "message": "running the three leakage checks against the index and probing canaries"})
            + "\n\n"
        )
        sc = await _scorecard_with_leakage(results, seeds)
        yield (
            "data: "
            + json.dumps({"type": "scorecard", "data": sc.model_dump(mode="json")})
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ScoreOneRequest(BaseModel):
    """Payload for POST /score-one."""

    question: str
    answer: str
    contexts: list[str] = []
    strategy: str = "seed"
    expected_behaviour: str = "match"
    # Re-pin the primary judge's model for this call.
    judge_model: str | None = None


@app.post("/score-one")
async def score_one(req: ScoreOneRequest) -> dict[str, Any]:
    """Score a single question-answer pair with all available judges.

    Both OpenAI judges (nano primary + mini secondary) score every request so
    the UI always shows per-judge verdicts for comparison. Consistent with the
    demo suite.
    """
    judges = available_judges(judge_model=req.judge_model)
    if not judges:
        raise HTTPException(
            status_code=503,
            detail="No judges available - set OPENAI_API_KEY in .env.",
        )

    h = hashlib.sha1(req.question.encode()).hexdigest()[:8]
    try:
        strat = Strategy(req.strategy)
        exp = ExpectedBehaviour(req.expected_behaviour)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    case = AdversarialCase(
        id=f"manual-{h}",
        seed_id="manual",
        strategy=strat,
        prompt=req.question,
        expected_behaviour=exp,
    )
    response = SUTResponse(
        answer=req.answer,
        retrieved_contexts=req.contexts,
        refused=not req.answer.strip(),
    )
    verdicts = await score_with_all(judges, case, response)
    avg = sum(v.score for v in verdicts) / len(verdicts) if verdicts else 0.0
    return {
        "question":    req.question,
        "answer":      req.answer,
        "strategy":    req.strategy,
        "avg_score":   round(avg, 2),
        "judge_count": len(verdicts),
        "verdicts":    [v.model_dump() for v in verdicts],
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
