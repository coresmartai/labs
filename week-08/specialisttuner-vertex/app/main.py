"""FastAPI surface for the SpecialistTuner (Vertex Gemini) app.

Routes:
  GET  /                        Browser UI
  GET  /health                  Liveness probe + pinned model
  GET  /readme                  README rendered as HTML
  POST /predict                 Run inference (base or tuned endpoint)
  POST /convert                 Generic JSONL → Vertex JSONL
  POST /v1/tune/start           Kick off Vertex supervised tuning
  GET  /v1/tune/status?job_id=  Poll a running tuning job
  GET  /v1/tuned-models         List tuning jobs in the project
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response

from app.config import get_settings
from app.llm import generate
from app.tools import execute_tool
from app.schemas import (
    ConvertRequest, ConvertResponse, InferenceRequest, InferenceResponse,
    TuneStartRequest,
)

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("specialisttuner_vertex")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = get_settings()  # validate env on startup
    logger.info("startup source_model=%s location=%s", settings.source_model, settings.gcp_location)
    yield


app = FastAPI(title="SpecialistTuner (Vertex Gemini)", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {
        "status": "ok",
        "provider": "vertex-ai",
        "model": s.source_model,
        "location": s.gcp_location,
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """This track has no browser page of its own, on purpose.

    The local package ships a UI because the training runs on your machine and
    there is something to watch. Here the job runs in somebody else's data
    centre, and their console is the surface for it. Read the README, drive the
    API, and watch the job in the Vertex AI console.
    """
    return RedirectResponse(url="/readme")


@app.get("/readme", include_in_schema=False)
def serve_readme() -> Response:
    import markdown as _md
    readme = Path(__file__).parent.parent / "README.md"
    if not readme.exists():
        raise HTTPException(status_code=404, detail="README.md not found")
    body = _md.markdown(
        readme.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>"
        "body{background:#0d1117;color:#e6edf3;font-family:sans-serif;"
        "max-width:900px;margin:40px auto;padding:0 20px;line-height:1.6}"
        "a{color:#58a6ff}code{background:#161b22;padding:2px 6px;border-radius:4px}"
        "pre{background:#161b22;padding:16px;border-radius:8px;overflow-x:auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #30363d;padding:8px 12px}"
        "h1,h2,h3{border-bottom:1px solid #30363d;padding-bottom:6px}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )
    return Response(content=page, media_type="text/html; charset=utf-8")


# ─── Inference ────────────────────────────────────────────────────────

@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest, tuned_endpoint: str | None = None) -> InferenceResponse:
    """Generate against either the base model or a tuned endpoint.

    Pass ?tuned_endpoint=projects/.../endpoints/... to hit a tuned model;
    omit for the base source model (used for the SpecialistTuner baseline
    row in the decision memo).
    """

    try:
        result = generate(
            req.prompt,
            tuned_endpoint=tuned_endpoint,
            temperature=req.temperature,
            max_output_tokens=req.max_output_tokens,
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return InferenceResponse(
        output=result["output"],
        model_id=result["model_id"],
        latency_ms=result["latency_ms"],
    )


# ─── Convert generic JSONL → Vertex JSONL ─────────────────────────────

@app.post("/convert", response_model=ConvertResponse)
def convert_route(req: ConvertRequest) -> ConvertResponse:

    result = execute_tool("convert", {
        "generic_path": req.generic_path,
        "gcloud_path": req.gcloud_path,
    })
    if result.get("success") is False:
        raise HTTPException(status_code=400, detail=result.get("error", "convert_failed"))
    return ConvertResponse(
        input_rows=result["input_rows"],
        output_rows=result["output_rows"],
        output_path=result["output_path"],
        warnings=result.get("warnings", []),
    )


# ─── Tuning ───────────────────────────────────────────────────────────

@app.post("/v1/tune/start")
def tune_start(req: TuneStartRequest) -> dict[str, Any]:
    """Kick off a supervised tuning job. Returns immediately with job_id."""

    args = {k: v for k, v in {
        "train_dataset_uri": req.train_dataset_uri,
        "epochs": req.epochs,
        "adapter_size": req.adapter_size,
        "learning_rate_multiplier": req.learning_rate_multiplier,
    }.items() if v is not None}
    result = execute_tool("start_tune", args)
    if result.get("success") is False:
        raise HTTPException(status_code=500, detail=result.get("error", "start_tune_failed"))
    return result


@app.get("/v1/tune/status")
def tune_status(job_id: str) -> dict[str, Any]:
    """Poll a running tuning job. Blocks briefly; the browser UI polls this."""
    # sleep_s=0 → don't loop; the client polls at its own cadence
    result = execute_tool("poll_tune", {"job_id": job_id, "sleep_s": 0, "max_wait_s": 0})
    if "state" not in result:
        raise HTTPException(status_code=404, detail=result.get("error", "unknown_job"))
    return result


@app.get("/v1/tuned-models")
def list_tuned() -> dict[str, Any]:
    """List all supervised tuning jobs in this project + location."""
    return execute_tool("list_tuned_models", {})
