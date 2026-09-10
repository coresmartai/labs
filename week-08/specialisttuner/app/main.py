"""FastAPI app that serves the fine-tuned LoRA adapter behind a small REST surface.

Routes:
  GET  /                         Browser UI (index.html)
  GET  /health                   Liveness probe
  GET  /readme                   README rendered as dark HTML
  POST /v1/chat/completions      LoRA inference
  POST /v1/fine-tune/start       Begin background LoRA training
  GET  /v1/fine-tune/status      Training progress (step, loss, epoch)
  POST /v1/fine-tune/stop        Signal training to stop
  GET  /v1/checkpoints           List adapter/checkpoint dirs under out/
  POST /v1/eval/run              Run eval_aiayn.jsonl against the model
  GET  /v1/eval/results          Last eval results (incl. the memo row)

Every eval run also emits one `EvalResult` row - the unit the SpecialistTuner
decision memo is built from. This module produces the `base` and `lora` rows
against the 42-question eval set.
"""
from __future__ import annotations
import json
import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, ConfigDict

from app.config import get_settings
from app.llm import generate
from app.schemas import ChatMessage, EvalResult, InferenceRequest, InferenceResponse

# Trainable-parameter count for the shipped LoRA config on Qwen3-0.6B:
# r=8 on q/k/v/o -> 81,920 per layer x 28 layers. Printed by
# `wrap_with_lora()` at train time; restated here so an eval run can emit a
# complete memo row without loading the model.
LORA_TRAINABLE_PARAMS = 2_293_760


settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("specialisttuner_lora")

app = FastAPI(title="SpecialistTuner LoRA Inference", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Inline request models for new routes
# ---------------------------------------------------------------------------

class _TrainReq(BaseModel):
    dataset_path: str = "data/specialisttuner.jsonl"
    r: int = 8
    alpha: int = 16
    epochs: int = 4
    lr: float = 2e-4


class _EvalReq(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    eval_path: str = "data/eval_aiayn.jsonl"
    adapter_path: str | None = "out/adapter"


# ---------------------------------------------------------------------------
# Shared mutable state (updated from background threads)
# ---------------------------------------------------------------------------

_train_state: dict[str, Any] = {
    "status": "idle",          # idle | starting | running | done | error
    "step": 0,
    "total": 0,
    "epoch": 0,
    "epochs": 0,
    "train_loss": [],          # [{"step": int, "loss": float}, ...]
    "eval_loss": [],           # [{"epoch": float, "loss": float}, ...]
    "trainable_params": 0,
    "error": None,
    "stop_flag": False,
}
_eval_results: dict[str, Any] = {}
_train_thread: threading.Thread | None = None


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------

def _run_training(req: _TrainReq) -> None:
    """Runs inside a daemon thread - updates _train_state in place."""
    global _train_state
    try:
        # All GPU imports are lazy so the module loads without CUDA
        from transformers import TrainerCallback  # type: ignore[import]
        from app.tools import (
            load_base_model,
            wrap_with_lora,
            load_and_tokenize_dataset,
            make_trainer,
        )

        class _ProgressCB(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if _train_state["stop_flag"]:
                    control.should_training_stop = True
                _train_state["step"] = state.global_step
                _train_state["total"] = int(state.max_steps or 0)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                if "loss" in logs:
                    _train_state["train_loss"].append(
                        {"step": state.global_step, "loss": round(float(logs["loss"]), 4)}
                    )
                if "eval_loss" in logs:
                    _train_state["eval_loss"].append(
                        {"epoch": round(float(state.epoch or 0), 2),
                         "loss": round(float(logs["eval_loss"]), 4)}
                    )

            def on_epoch_end(self, args, state, control, **kwargs):
                _train_state["epoch"] = int(state.epoch or 0)

        _train_state["status"] = "running"
        logger.info("Training thread started r=%d alpha=%d epochs=%d lr=%s",
                    req.r, req.alpha, req.epochs, req.lr)

        model = load_base_model()
        model = wrap_with_lora(model, r=req.r, alpha=req.alpha)
        _train_state["trainable_params"] = int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
        dataset = load_and_tokenize_dataset(Path(req.dataset_path))
        trainer = make_trainer(model, dataset, Path("out"), epochs=req.epochs, lr=req.lr)
        trainer.add_callback(_ProgressCB())
        trainer.train()
        _train_state["status"] = "done"
        logger.info("Training done")
    except Exception as exc:
        _train_state["status"] = "error"
        _train_state["error"] = str(exc)
        logger.exception("Training failed")


# ---------------------------------------------------------------------------
# Existing routes (unchanged)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> HTMLResponse:
    f = Path(__file__).parent.parent / "index.html"
    return HTMLResponse(content=f.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    s = get_settings()
    return {"status": "ok", "model": f"{s.base_model}+lora"}


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
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>README - SpecialistTuner LoRA</title><style>"
        "body{background:#0d1117;color:#e6edf3;font-family:sans-serif;"
        "max-width:900px;margin:40px auto;padding:0 20px;line-height:1.7}"
        "h1,h2,h3{color:#e6edf3;border-bottom:1px solid #30363d;padding-bottom:.3em}"
        "code{background:#21262d;padding:2px 6px;border-radius:4px;font-size:.9em}"
        "pre{background:#161b22;padding:16px;border-radius:8px;overflow-x:auto}"
        "pre code{background:none;padding:0}a{color:#58a6ff}"
        "table{border-collapse:collapse;width:100%;margin:16px 0}"
        "th,td{border:1px solid #30363d;padding:8px 12px;text-align:left}"
        "th{background:#161b22}"
        "blockquote{border-left:4px solid #30363d;margin:0;padding:0 16px;color:#7d8590}"
        "</style></head>"
        f"<body>{body}</body></html>"
    )
    return Response(content=page, media_type="text/html; charset=utf-8")


@app.post("/v1/chat/completions", response_model=InferenceResponse)
def chat(req: InferenceRequest, adapter: str | None = None) -> InferenceResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages required")
    try:
        result = generate(
            req.messages,
            adapter_path=adapter,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
        )
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))
    s = get_settings()
    return InferenceResponse(
        output=result["output"],
        model_id=f"{s.base_model}+lora",
        latency_ms=result["latency_ms"],
    )

# ---------------------------------------------------------------------------
# Fine-tune routes
# ---------------------------------------------------------------------------

@app.post("/v1/fine-tune/start")
def start_training(req: _TrainReq) -> dict[str, Any]:
    """Kick off background LoRA training. Returns immediately."""
    global _train_thread
    if _train_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already running")
    # Reset state
    _train_state.update({
        "status": "starting",
        "step": 0, "total": 0, "epoch": 0, "epochs": req.epochs,
        "train_loss": [], "eval_loss": [],
        "trainable_params": 0, "error": None, "stop_flag": False,
    })
    _train_thread = threading.Thread(target=_run_training, args=(req,), daemon=True)
    _train_thread.start()
    return {"message": "Training started", "config": req.model_dump()}


@app.get("/v1/fine-tune/status")
def training_status() -> dict[str, Any]:
    """Current training progress. Poll this at 2s intervals while status==running."""
    return dict(_train_state)


@app.post("/v1/fine-tune/stop")
def stop_training() -> dict[str, str]:
    """Send a stop signal; training stops at the next step boundary."""
    if _train_state["status"] != "running":
        return {"message": "No training in progress"}
    _train_state["stop_flag"] = True
    return {"message": "Stop signal sent"}


@app.get("/v1/checkpoints")
def list_checkpoints() -> list[str]:
    """List adapter / checkpoint directories under the configured output_dir.

    Returns paths relative to the server cwd, e.g. ["out/adapter", "out/checkpoint-164"].
    The UI uses these to populate the adapter dropdown - no manual typing needed.
    """
    s = get_settings()
    out_dir = Path(s.output_dir)
    if not out_dir.exists():
        return []
    return sorted(
        str(p).replace("\\", "/")
        for p in out_dir.iterdir()
        if p.is_dir()
    )


# ---------------------------------------------------------------------------
# Eval routes
# ---------------------------------------------------------------------------

@app.post("/v1/eval/run")
def run_eval(req: _EvalReq) -> dict[str, Any]:
    """Run eval_aiayn.jsonl against the model and return scored results."""
    eval_path = Path(req.eval_path)
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail=f"Eval file not found: {req.eval_path}")

    rows: list[dict] = []
    with eval_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results: list[dict] = []
    for row in rows:
        question: str = row["question"]
        expected: str = row["expected"]
        accept: list[str] = row.get("accept", [expected])
        category: str = row.get("category", "general")

        try:
            result = generate(
                [ChatMessage(role="user", content=question)],
                adapter_path=req.adapter_path,
                max_new_tokens=64,
                temperature=0.0,
            )
            got: str = result["output"].strip()
            got_lower = got.lower()
            passed = any(a.lower() in got_lower for a in accept)
            results.append({
                "id": row.get("id"),
                "question": question,
                "expected": expected,
                "got": got,
                "passed": passed,
                "category": category,
                "latency_ms": result["latency_ms"],
            })
        except Exception as exc:
            results.append({
                "id": row.get("id"),
                "question": question,
                "expected": expected,
                "got": f"ERROR: {exc}",
                "passed": False,
                "category": category,
                "latency_ms": 0.0,
            })

    # Aggregate
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    by_cat: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"total": 0, "passed": 0}
        by_cat[cat]["total"] += 1
        if r["passed"]:
            by_cat[cat]["passed"] += 1

    cat_accuracy = {
        cat: round(v["passed"] / v["total"] * 100, 1)
        for cat, v in by_cat.items()
    }

    accuracy = round(passed_count / total * 100, 1) if total else 0.0
    _eval_results.update({
        "total": total,
        "passed": passed_count,
        "accuracy": accuracy,
        "by_category": cat_accuracy,
        "results": results,
        "row": _memo_row(req.adapter_path, accuracy, results).model_dump(),
    })
    return dict(_eval_results)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy - this is the only stat we need."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * len(ordered) + 0.5) - 1))
    return round(ordered[idx], 1)


def _memo_row(adapter_path: str | None, accuracy: float,
              results: list[dict]) -> EvalResult:
    """Build the one `EvalResult` row this eval run contributes to the memo.

    Nothing here hard-codes a row count: the memo is however many rows the
    harness is handed.
    """
    s = get_settings()
    lats = [float(r["latency_ms"]) for r in results if r["latency_ms"]]
    tuned = bool(adapter_path)
    return EvalResult(
        approach="lora" if tuned else "base",
        model_id=f"{s.base_model}+lora" if tuned else s.base_model,
        accuracy=accuracy,
        p50_latency_ms=_percentile(lats, 50),
        p95_latency_ms=_percentile(lats, 95),
        # Local run on hardware you already own: no per-call vendor bill.
        cost_per_1k_calls_usd=0.0,
        trainable_params=LORA_TRAINABLE_PARAMS if tuned else 0,
        # The weights never leave the machine - that is the whole open-path case.
        privacy_posture="in_perimeter",
    )


@app.get("/v1/eval/results")
def eval_results() -> dict[str, Any]:
    """Return the most recent eval run results."""
    if not _eval_results:
        return {"message": "No eval run yet - POST /v1/eval/run first"}
    return dict(_eval_results)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def run() -> None:
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.model_host, port=s.model_port, reload=False)


if __name__ == "__main__":
    run()
