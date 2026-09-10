"""SpecialistTuner Vertex AI pipeline. Six composable functions + dispatcher.

The pattern matches W8V2 and W10V3: no agent framework, no orchestration
library. A plain dict maps names to callables, and a small dispatcher
looks them up and calls them with a kwargs dict.

Pipeline in order:
  1. validate_generic   → lint the generic JSONL
  2. convert            → generic JSONL → Vertex JSONL
  3. upload             → local Vertex JSONL → GCS URI
  4. start_tune         → kick off supervised tuning against a GCS URI
  5. poll_tune          → follow a running tuning job until terminal state
  6. list_tuned_models  → enumerate finished tuned models (cleanup input)

Every heavy import (google-cloud-*, vertexai) is deferred inside the
functions so the module can be imported on a machine without the SDK
present, and that is what keeps the offline smoke tests fast.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Callable

from app.config import get_settings
from app.convert import convert_file, validate_generic
from prompts import load_system_instruction

log = logging.getLogger(__name__)


# ─── 1. Validate the generic file ─────────────────────────────────────

def validate(path: str | Path | None = None) -> dict[str, Any]:
    """Lint the generic JSONL. Always run this before you burn quota."""
    s = get_settings()
    p = Path(path) if path else s.generic_dataset_path
    return validate_generic(p)


# ─── 2. Convert generic → Vertex JSONL ────────────────────────────────

def convert(generic_path: str | Path | None = None,
            gcloud_path: str | Path | None = None,
            include_system_instruction: bool = True) -> dict[str, Any]:
    """Convert local generic JSONL to Vertex-shaped local JSONL.

    The system instruction is included by default so every training row
    carries the teacher persona at the top level of the record, and the
    Vertex tuning job uses that during training and it's what the model
    then absorbs into its parameters.
    """
    s = get_settings()
    gp = Path(generic_path) if generic_path else s.generic_dataset_path
    cp = Path(gcloud_path) if gcloud_path else s.gcloud_dataset_path
    system_instruction = load_system_instruction() if include_system_instruction else None
    return convert_file(gp, cp, system_instruction=system_instruction)


# ─── 3. Upload local file to GCS ──────────────────────────────────────

def upload_to_gcs(local_path: str | Path | None = None,
                  prefix: str | None = None) -> dict[str, Any]:
    """Upload a local file to the configured GCS bucket and return its
    gs:// URI. The tuning job needs a GCS URI, not a local path."""
    from google.cloud import storage

    s = get_settings()
    local_path = Path(local_path) if local_path else s.gcloud_dataset_path
    if not local_path.exists():
        return {"success": False, "error": "file_not_found", "path": str(local_path)}

    bucket_name = s.gcs_bucket.replace("gs://", "").split("/", 1)[0]
    key_prefix = prefix or s.gcs_train_prefix
    blob_name = f"{key_prefix}/{local_path.name}"

    client = storage.Client(project=s.gcp_project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))

    uri = f"gs://{bucket_name}/{blob_name}"
    log.info("upload_to_gcs %s → %s", local_path, uri)
    return {"success": True, "uri": uri, "bytes": local_path.stat().st_size}


# ─── 4. Kick off supervised tuning ────────────────────────────────────

def start_tune(train_dataset_uri: str | None = None,
               epochs: int | None = None,
               adapter_size: int | None = None,
               learning_rate_multiplier: float | None = None,
               display_name: str | None = None) -> dict[str, Any]:
    """Start a Vertex Gemini supervised-tuning job.

    Returns the job resource name plus the initial state. The job runs
    server-side; call poll_tune() to wait for it or hit the tuning-jobs
    UI in the Google Cloud console.
    """
    from vertexai.tuning import sft
    from app.llm import init_vertex

    init_vertex()
    s = get_settings()
    job = sft.train(
        source_model=s.source_model,
        train_dataset=train_dataset_uri or f"{s.gcs_bucket.rstrip('/')}/{s.gcs_train_prefix}/aiayn_gcloud.jsonl",
        epochs=epochs if epochs is not None else s.tuning_epochs,
        adapter_size=adapter_size if adapter_size is not None else s.tuning_adapter_size,
        learning_rate_multiplier=(
            learning_rate_multiplier if learning_rate_multiplier is not None
            else s.learning_rate_multiplier
        ),
        tuned_model_display_name=display_name or s.tuned_model_display_name,
    )
    log.info("start_tune job=%s state=%s", job.resource_name, job.state)
    return {
        "success": True,
        "job_id": job.resource_name,
        "state": str(job.state),
        "display_name": display_name or s.tuned_model_display_name,
    }


# ─── 5. Poll a tuning job until terminal ──────────────────────────────

def poll_tune(job_id: str, sleep_s: float = 60.0,
              max_wait_s: float = 4 * 3600) -> dict[str, Any]:
    """Poll a tuning job until it reaches a terminal state.

    Terminal states: SUCCEEDED, FAILED, CANCELLED, EXPIRED. On success
    returns the tuned model endpoint resource name so the caller can
    hand it to app.llm.generate() as the `tuned_endpoint` argument.
    """
    from vertexai.tuning import sft
    from app.llm import init_vertex

    init_vertex()
    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    t0 = time.monotonic()
    while True:
        job = sft.SupervisedTuningJob(job_id)
        state = str(job.state)
        if state in terminal:
            endpoint = getattr(job, "tuned_model_endpoint_name", None)
            log.info("poll_tune job=%s reached %s endpoint=%s", job_id, state, endpoint)
            return {
                "success": state == "JOB_STATE_SUCCEEDED",
                "state": state,
                "tuned_model_endpoint": endpoint,
                "error": getattr(job, "error", None),
            }
        if (time.monotonic() - t0) > max_wait_s:
            return {"success": False, "state": state, "error": "timeout"}
        log.info("poll_tune job=%s state=%s (waiting)", job_id, state)
        time.sleep(sleep_s)


# ─── 6. List finished tuned models (used by cleanup runbook) ──────────

def list_tuned_models() -> dict[str, Any]:
    """List all supervised tuning jobs in this project + location."""
    from vertexai.tuning import sft
    from app.llm import init_vertex

    init_vertex()
    jobs = list(sft.SupervisedTuningJob.list())
    return {
        "success": True,
        "count": len(jobs),
        "jobs": [
            {
                "job_id": j.resource_name,
                "state": str(j.state),
                "display_name": getattr(j, "display_name", None),
                "endpoint": getattr(j, "tuned_model_endpoint_name", None),
            }
            for j in jobs
        ],
    }


# ─── Tiny dispatcher ──────────────────────────────────────────────────

TOOLS: dict[str, Callable[..., Any]] = {
    "validate":          validate,
    "convert":           convert,
    "upload":            upload_to_gcs,
    "start_tune":        start_tune,
    "poll_tune":         poll_tune,
    "list_tuned_models": list_tuned_models,
}


def execute_tool(name: str, args: dict[str, Any] | None = None) -> Any:
    """Dispatch a named tool. Never raises; unknown/failure returns dict."""
    args = args or {}
    if name not in TOOLS:
        return {"success": False, "error": "unknown_tool", "tool": name,
                "available": sorted(TOOLS.keys())}
    try:
        return TOOLS[name](**args)
    except Exception as exc:  # noqa: BLE001
        log.exception("execute_tool %s failed", name)
        return {"success": False, "error": str(exc), "tool": name}
