"""
FastAPI entry point + CLI runner.

Endpoints:
  GET  /                    - serve Web UI (index.html)
  GET  /health              - liveness + configured providers
  POST /classify            - classify a single message via a chosen provider
  POST /benchmark           - run the full golden-set benchmark, return summaries (sync)
  GET  /benchmark-stream    - same benchmark streamed as SSE (used by the Web UI)

CLI:
  python -m app.main benchmark        - runs benchmark and prints the table
  uvicorn app.main:app --reload       - serve the HTTP API + Web UI
"""
from __future__ import annotations
import asyncio
import json
import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import get_settings
from app.eval import load_golden, run_benchmark, run_one, summarise, write_csv
from app.schemas import BenchmarkRow, BenchmarkSummary, ClassifyRequest, Result


settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("intentiq")

app = FastAPI(
    title="IntentIQ",
    version="0.1.0",
    description="Week 2 - multi-provider intent classifier + golden-dataset benchmark.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GOLDEN_PATH = Path(__file__).parent.parent / "golden_dataset.jsonl"
UI_PATH     = Path(__file__).parent.parent / "index.html"


# --------------------------------------------------------------------------- #
# Web UI                                                                      #
# --------------------------------------------------------------------------- #
@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    """Render README.md as a dark-themed HTML page - linked from the Web UI header."""
    import markdown as _md

    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        raise HTTPException(status_code=404, detail="README.md not found")

    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>README - IntentIQ</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0d1117; color: #e6edf3;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 15px; line-height: 1.7;
      padding: 40px 20px;
    }}
    .page {{ max-width: 860px; margin: 0 auto; }}
    h1, h2, h3, h4 {{ color: #e6edf3; font-weight: 600; margin: 1.4em 0 0.5em; line-height: 1.3; }}
    h1 {{ font-size: 2em; border-bottom: 1px solid #30363d; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.4em; border-bottom: 1px solid #21262d; padding-bottom: 0.2em; }}
    h3 {{ font-size: 1.1em; }}
    p  {{ margin: 0.7em 0; color: #c9d1d9; }}
    a  {{ color: #58a6ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{
      background: #161b22; border: 1px solid #30363d;
      border-radius: 4px; padding: 2px 6px;
      font-family: 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
      font-size: 0.88em; color: #79c0ff;
    }}
    pre {{
      background: #161b22; border: 1px solid #30363d;
      border-radius: 8px; padding: 16px; overflow-x: auto;
      margin: 1em 0;
    }}
    pre code {{
      border: none; padding: 0; background: none;
      font-size: 0.87em; color: #e6edf3;
    }}
    blockquote {{
      border-left: 3px solid #30363d; margin: 1em 0;
      padding: 6px 16px; color: #7d8590;
    }}
    table {{
      border-collapse: collapse; width: 100%; margin: 1em 0;
      font-size: 0.93em;
    }}
    th, td {{
      border: 1px solid #30363d; padding: 8px 14px; text-align: left;
    }}
    th {{ background: #161b22; color: #7d8590; font-weight: 600;
          text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.04em; }}
    tr:nth-child(even) td {{ background: #0d1117; }}
    tr:nth-child(odd)  td {{ background: #161b22; }}
    ul, ol {{ padding-left: 1.6em; margin: 0.6em 0; color: #c9d1d9; }}
    li {{ margin: 0.25em 0; }}
    hr {{ border: none; border-top: 1px solid #30363d; margin: 2em 0; }}
    strong {{ color: #e6edf3; }}
  </style>
</head>
<body>
  <div class="page">
    {body}
  </div>
</body>
</html>"""

    return Response(content=html, media_type="text/html; charset=utf-8")


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    """Serve the Web UI - open http://localhost:8000 in any browser."""
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found next to app/")
    return FileResponse(UI_PATH, media_type="text/html")


# --------------------------------------------------------------------------- #
# Health                                                                      #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    """
    Liveness check - returns server status, all three model pins, and which
    providers are currently configured (have API keys set in .env).
    The UI health chip reads the 'model' key from this response.
    """
    s = get_settings()
    configured = []
    if s.openai_api_key:
        configured.extend(["openai", "nano"])
    configured.append("ollama")          # always listed - local, no key needed
    return {
        "status": "ok",
        "model": s.openai_model,         # read by the shared UI checkHealth() function
        "openai_model": s.openai_model,
        "nano_model": s.nano_model,
        "ollama_model": s.ollama_model,
        "ollama_base_url": s.ollama_base_url,
        "providers_configured": configured,
    }


# --------------------------------------------------------------------------- #
# Debug - provider connectivity smoke-test                                    #
# --------------------------------------------------------------------------- #
@app.get("/debug/providers", include_in_schema=False)
def debug_providers() -> dict:
    """
    Quick smoke-test for all three providers.
    Open http://localhost:8000/debug/providers in your browser.
    Not included in the OpenAPI schema - dev use only.
    """
    from openai import APIError, OpenAI

    s       = get_settings()
    results: dict = {}

    # ── OpenAI providers ─────────────────────────────────────────────────────
    if not s.openai_api_key:
        for p in ("openai", "nano"):
            results[p] = {"ok": False, "error": "OPENAI_API_KEY not set in .env"}
    else:
        oa_client = OpenAI(api_key=s.openai_api_key)
        for name, model in [("openai", s.openai_model), ("nano", s.nano_model)]:
            try:
                resp = oa_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with one word: hello"}],
                    max_completion_tokens=20,
                )
                results[name] = {"ok": True, "model": model,
                                 "reply": resp.choices[0].message.content}
            except APIError as exc:
                results[name] = {"ok": False, "model": model,
                                 "error": f"HTTP {getattr(exc,'status_code','?')}: "
                                          f"{getattr(exc,'message',None) or str(exc)}"}
            except Exception as exc:  # noqa: BLE001
                results[name] = {"ok": False, "model": model, "error": str(exc)}

    # ── Ollama (local) ───────────────────────────────────────────────────────
    try:
        ol_client = OpenAI(api_key="ollama", base_url=s.ollama_base_url)
        resp = ol_client.chat.completions.create(
            model=s.ollama_model,
            messages=[{"role": "user", "content": "Reply with one word: hello"}],
            max_tokens=20,
            temperature=0.0,
        )
        results["ollama"] = {"ok": True, "model": s.ollama_model,
                             "reply": resp.choices[0].message.content}
    except APIError as exc:
        results["ollama"] = {"ok": False, "model": s.ollama_model,
                             "error": f"HTTP {getattr(exc,'status_code','?')}: "
                                      f"{getattr(exc,'message',None) or str(exc)}"}
    except Exception as exc:  # noqa: BLE001
        results["ollama"] = {"ok": False, "model": s.ollama_model,
                             "error": f"{type(exc).__name__}: {exc}"}

    return {
        "ollama_base_url": s.ollama_base_url,
        "providers": results,
    }


# --------------------------------------------------------------------------- #
# Classify                                                                    #
# --------------------------------------------------------------------------- #
@app.post("/classify", response_model=Result)
def classify_endpoint(req: ClassifyRequest) -> Result:
    """
    Classify a single message using the requested provider.

    Request body: { "text": "...", "provider": "openai" | "nano" | "ollama" }
    Returns a unified Result object regardless of which provider handled the call.
    Raises HTTP 400 if the provider key is not configured or the API call fails.
    Raises HTTP 422 if the provider value is not one of the valid literals.
    """
    from app.llm import classify
    try:
        return classify(req.provider, req.text)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------------------------------- #
# Benchmark - sync (CLI / notebook / curl)                                    #
# --------------------------------------------------------------------------- #
@app.post("/benchmark", response_model=list[BenchmarkSummary])
def benchmark_endpoint() -> list[BenchmarkSummary]:
    """
    Run the full golden-dataset benchmark synchronously.
    Calls every configured provider on all 30 examples and returns per-provider
    summaries (accuracy, p50/p95 latency, cost per 1K calls).
    Also writes results.csv next to golden_dataset.jsonl.
    Use /benchmark-stream instead when calling from the Web UI (live progress).
    """
    if not GOLDEN_PATH.exists():
        raise HTTPException(status_code=500, detail=f"golden_dataset.jsonl not found at {GOLDEN_PATH}")
    rows = run_benchmark(load_golden(GOLDEN_PATH))
    write_csv(rows, Path(__file__).parent.parent / "results.csv")
    return summarise(rows)


# --------------------------------------------------------------------------- #
# Benchmark - streaming SSE (Web UI)                                          #
#                                                                             #
# Event types emitted:                                                        #
#   start          {providers, n}                                             #
#   provider_start {provider}                                                 #
#   progress       {provider, done, total, correct, accuracy, avg_lat_ms}     #
#   provider_done  {provider, summary}                                        #
#   done           {summaries}                                                #
# --------------------------------------------------------------------------- #
@app.get("/benchmark-stream", include_in_schema=False)
async def benchmark_stream_endpoint() -> StreamingResponse:
    """
    Stream the benchmark as Server-Sent Events so the Web UI can show live
    progress bars and partial results as each provider completes.

    SSE event types:
      start         - {providers, n}           fired once at the beginning
      provider_start - {provider}              fired when a provider's loop begins
      progress      - {provider, done, total, correct, accuracy, avg_lat_ms}
      provider_done - {provider, summary}      fired when all examples for a provider finish
      done          - {summaries}              fired after all providers complete
    """
    if not GOLDEN_PATH.exists():
        raise HTTPException(status_code=500, detail=f"golden_dataset.jsonl not found at {GOLDEN_PATH}")

    cfg    = get_settings()
    golden = load_golden(GOLDEN_PATH)
    n      = len(golden)

    active: list[str] = []
    if cfg.openai_api_key:
        active.extend(["openai", "nano"])
    active.append("ollama")          # always included - local, no key needed

    def _sse(data: dict) -> str:
        """Serialise *data* to a single SSE frame string (data: ...\n\n)."""
        return f"data: {json.dumps(data)}\n\n"

    async def _generate():
        """Async generator that yields SSE frames for each benchmark event."""
        yield _sse({"type": "start", "providers": active, "n": n})

        all_rows: list[BenchmarkRow] = []

        for provider in active:
            yield _sse({"type": "provider_start", "provider": provider})
            prows: list[BenchmarkRow] = []

            for ex in golden:
                try:
                    row: BenchmarkRow = await asyncio.to_thread(run_one, provider, ex)
                except Exception:
                    row = BenchmarkRow(
                        example_id=ex.id, provider=provider,
                        predicted="unknown", expected=ex.expected,
                        correct=False, confidence=0.0,
                        latency_ms=0.0, input_tokens=0, output_tokens=0,
                    )
                prows.append(row)
                done    = len(prows)
                correct = sum(1 for r in prows if r.correct)
                warm    = [r.latency_ms for r in prows if r.latency_ms > 0]
                yield _sse({
                    "type":       "progress",
                    "provider":   provider,
                    "done":       done,
                    "total":      n,
                    "correct":    correct,
                    "accuracy":   correct / done,
                    "avg_lat_ms": sum(warm) / len(warm) if warm else 0.0,
                })

            all_rows.extend(prows)
            psummaries = summarise(prows)
            yield _sse({
                "type":    "provider_done",
                "provider": provider,
                "summary": psummaries[0].model_dump() if psummaries else None,
            })

        write_csv(all_rows, Path(__file__).parent.parent / "results.csv")
        yield _sse({
            "type":      "done",
            "summaries": [smry.model_dump() for smry in summarise(all_rows)],
        })
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _print_summary(summaries: list[BenchmarkSummary]) -> None:
    """Print a formatted benchmark results table to stdout for the CLI runner."""
    print(f"\n{'provider':<11}{'n':>4}{'acc':>8}{'p50':>10}{'p95':>10}{'$/1K':>10}")
    print("-" * 53)
    for s in summaries:
        print(
            f"{s.provider:<11}{s.n:>4}{s.accuracy*100:>7.1f}%"
            f"{s.p50_ms:>9.0f}ms{s.p95_ms:>9.0f}ms{s.cost_per_1k_usd:>9.4f}"
        )


def cli_benchmark() -> None:
    """
    CLI entry point: run the full benchmark without starting the HTTP server.
    Invoked via: python -m app.main benchmark
    Loads the golden dataset, runs all configured providers, prints the summary
    table to stdout, and writes results.csv next to golden_dataset.jsonl.
    """
    logger.info("Loading golden dataset from %s", GOLDEN_PATH)
    if not GOLDEN_PATH.exists():
        logger.error("golden_dataset.jsonl missing at %s", GOLDEN_PATH)
        sys.exit(1)
    golden = load_golden(GOLDEN_PATH)
    logger.info("Running benchmark on %d examples...", len(golden))
    rows = run_benchmark(golden)
    write_csv(rows, Path(__file__).parent.parent / "results.csv")
    _print_summary(summarise(rows))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        cli_benchmark()
    else:
        print(json.dumps({"hint": "run `python -m app.main benchmark` or start with uvicorn."}))
