"""Thin wrapper around the Vertex AI SDK.

Route every model call through this module so the rest of the codebase
remains provider-agnostic. Heavy imports (vertexai, google.cloud.*) are
lazy, deferred to function call time, so `from app import llm` works on
a machine without the Google Cloud stack installed, which is what makes
the smoke tests fast and offline.

Two public functions:
  init_vertex()  - one-shot initialisation of vertexai (idempotent)
  generate(prompt, tuned_endpoint, ...) - synchronous inference, returns
    a dict with `output` and `latency_ms`
"""
from __future__ import annotations
import logging
import time
from typing import Any

from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt,
    wait_exponential_jitter, before_sleep_log,
)

from app.config import get_settings
from prompts import load_system_instruction

log = logging.getLogger(__name__)

_INITIALISED = False


def init_vertex() -> None:
    """Initialise vertexai once per process. Idempotent."""
    global _INITIALISED
    if _INITIALISED:
        return
    import vertexai
    s = get_settings()
    vertexai.init(project=s.gcp_project, location=s.gcp_location)
    _INITIALISED = True
    log.info("vertex init project=%s location=%s", s.gcp_project, s.gcp_location)


# Transient failures worth retrying (imported lazily; see the decorator).
def _retryable_exception_types():
    from google.api_core.exceptions import (
        DeadlineExceeded, InternalServerError, ResourceExhausted, ServiceUnavailable,
    )
    return (DeadlineExceeded, InternalServerError, ResourceExhausted, ServiceUnavailable)


@retry(
    retry=retry_if_exception_type(Exception),  # narrowed at call time
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=8.0),
    before_sleep=before_sleep_log(log, logging.WARNING),
    reraise=True,
)
def _call_gemini(model_id: str, prompt: str,
                 temperature: float, max_output_tokens: int) -> tuple[str, float]:
    """Actual Vertex call, wrapped in tenacity retries.

    Returns (text, latency_ms). Any exception is reraised after the third
    attempt so the caller (FastAPI route, eval harness) can decide how to
    surface it.
    """
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    init_vertex()
    model = GenerativeModel(
        model_name=model_id,
        system_instruction=load_system_instruction(),
    )
    cfg = GenerationConfig(
        temperature=max(temperature, 1e-5),  # zero → sample-with-no-randomness ambiguity
        max_output_tokens=max_output_tokens,
    )
    t0 = time.perf_counter()
    resp = model.generate_content(prompt, generation_config=cfg)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    text = getattr(resp, "text", "") or ""
    return text, latency_ms


def generate(prompt: str, tuned_endpoint: str | None = None,
             temperature: float = 0.2, max_output_tokens: int = 512) -> dict[str, Any]:
    """Inference path used by the FastAPI route + eval harness.

    * `tuned_endpoint` is the resource name of the tuned model endpoint
      (e.g. "projects/.../locations/us-central1/endpoints/1234"). When
      None, we hit the source model directly, which is what the baseline
      row of the SpecialistTuner decision memo.
    """
    s = get_settings()
    model_id = tuned_endpoint or s.source_model
    text, latency_ms = _call_gemini(model_id, prompt, temperature, max_output_tokens)
    return {"output": text, "latency_ms": latency_ms, "model_id": model_id}
