"""
Provider abstraction - one interface, three backends.

Providers
─────────
  openai     → OpenAI API (gpt-5.4-mini-2026-03-17)    via openai SDK
  nano       → OpenAI API (gpt-5.4-nano-2026-03-17)    via openai SDK (same key, different model)
  ollama     → Ollama local server (qwen3:0.6b)        via OpenAI-compatible endpoint

Both OpenAI providers share the same API key and client - they differ only in
the model pin. Ollama uses a dummy api_key ("ollama") and the local base URL.

Failure modes handled by the adapters (each one a class of bug that would
otherwise silently corrupt the benchmark numbers):
  1. 429 rate limits      - OpenAI providers retry once with 0.5s back-off
                            (sleep is inside the latency timer - honest reporting)
  2. Schema drift         - each adapter normalises into the same Result
                            (case + type drift from smaller models is fixed by
                            normalise_label and _safe_json before the Result is built)
  3. JSON parse fallback  - small / local models may wrap JSON in prose;
                            three-stage extraction (strict → fenced → bare → "unknown")
  4. Case mismatch        - normalise_label() lowercases + strips and maps unknown
                            labels to "unknown" rather than failing the comparison silently
  5. Ollama offline       - surfaces as RuntimeError so run_benchmark logs a failed
                            row instead of recording a silent 0% accuracy figure
"""
from __future__ import annotations
import json
import logging
import re
import time
from typing import Any

from openai import APIError as OpenAIAPIError
from openai import OpenAI, RateLimitError as OpenAIRateLimit

from app.config import get_settings
from app.schemas import INTENT_LABELS, Provider, Result

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def normalise_label(raw: str) -> str:
    """Lowercase, strip, collapse spaces/dashes to underscores. Fall back to 'unknown'."""
    if not raw:
        return "unknown"
    candidate = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return candidate if candidate in INTENT_LABELS else "unknown"


def _safe_json(s: str) -> dict[str, Any]:
    """
    Parse *s* as JSON and return it if it is a dict, otherwise return {}.
    Never raises - all parse failures are silently swallowed so callers can
    try the next extraction strategy (fenced block, bare brace, etc.).
    """
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# Regex fallbacks for when a model wraps JSON in prose or fenced code blocks.
_FENCED_JSON = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON    = re.compile(r"(\{[^{}]*\})", re.DOTALL)


_SYSTEM_PROMPT = (
    "You are an intent classifier. Classify the user message into exactly one of: "
    + ", ".join(INTENT_LABELS)
    + ". Return JSON with two keys: intent (string) and confidence (number 0..1)."
)


# --------------------------------------------------------------------------- #
# OpenAI shared helper - used by both openai and nano                  #
# --------------------------------------------------------------------------- #
def _openai_call(text: str, provider: Provider, model: str) -> Result:
    """
    Call the OpenAI chat completions API with json_object response format.
    Both openai and nano use the same API key - only the model differs.
    """
    s = get_settings()
    if not s.openai_api_key:
        raise RuntimeError(f"OPENAI_API_KEY not configured - skipping {provider}.")
    client = OpenAI(api_key=s.openai_api_key)

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            max_completion_tokens=80,
        )
    except OpenAIRateLimit:
        # Failure mode 1: rate limit - sleep 0.5s, retry once.
        time.sleep(0.5)
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            max_completion_tokens=80,
        )
    except OpenAIAPIError as exc:
        status = getattr(exc, "status_code", "?")
        body   = getattr(exc, "message", None) or str(exc)
        raise RuntimeError(f"OpenAI {provider}/{model} → HTTP {status}: {body}") from exc
    latency_ms = (time.perf_counter() - t0) * 1000.0

    content = resp.choices[0].message.content or "{}"
    payload = _safe_json(content)
    return Result(
        provider=provider,
        label=normalise_label(payload.get("intent", "unknown")),
        confidence=float(payload.get("confidence", 0.5)),
        latency_ms=latency_ms,
        input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        raw={"content": content},
    )


def openai_classify(text: str) -> Result:
    """Classify *text* using OpenAI gpt-5.4-mini. Requires OPENAI_API_KEY in .env."""
    return _openai_call(text, "openai", get_settings().openai_model)


def nano_classify(text: str) -> Result:
    """Classify *text* using OpenAI gpt-5.4-nano. Same API key as openai_classify, smaller model."""
    return _openai_call(text, "nano", get_settings().nano_model)


# --------------------------------------------------------------------------- #
# Ollama - local OpenAI-compatible endpoint                                   #
# --------------------------------------------------------------------------- #
def ollama_classify(text: str) -> Result:
    """
    Classify via a local Ollama server.

    Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1.
    Any non-empty string works as the api_key - "ollama" is the convention.

    Small models (qwen3:0.6b) rarely produce clean JSON, so we use the same
    three-stage extraction: strict → fenced code block → bare {...}.
    If all three fail, we return label="unknown" and confidence=0.5.
    """
    s = get_settings()
    client = OpenAI(api_key="ollama", base_url=s.ollama_base_url)

    # Defence-in-depth against Qwen3's chain-of-thought mode:
    #
    #   Layer 1 - extra_body={"think": False}
    #     Asks Ollama to suppress the reasoning phase entirely.
    #     Works on Ollama ≥ 0.6.5 with Qwen3 models; silently ignored otherwise.
    #
    #   Layer 2 - max_tokens=400
    #     Gives the model enough room to finish both reasoning AND the JSON answer
    #     if thinking fires anyway (Qwen3 burns ~100-300 tokens on reasoning before
    #     writing the actual answer in message.content).
    #
    #   Layer 3 - reasoning fallback (below, after the call)
    #     If message.content is still empty (thinking filled the context or Ollama
    #     version doesn't support the flag), the answer is extracted from
    #     message.model_extra["reasoning"] via the same three-stage JSON parser.
    _no_think = {"think": False}

    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=s.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            max_tokens=400,
            temperature=0.1,
            extra_body=_no_think,
        )
    except OpenAIRateLimit:
        time.sleep(0.5)
        resp = client.chat.completions.create(
            model=s.ollama_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            max_tokens=400,
            temperature=0.0,
            extra_body=_no_think,
        )
    except OpenAIAPIError as exc:
        status = getattr(exc, "status_code", "?")
        body   = getattr(exc, "message", None) or str(exc)
        raise RuntimeError(
            f"Ollama {s.ollama_model} → HTTP {status}: {body}"
            " - is Ollama running? Start with: ollama serve"
        ) from exc
    latency_ms = (time.perf_counter() - t0) * 1000.0

    msg     = resp.choices[0].message
    content = msg.content or ""

    # Layer 3: if content is still empty, fall back to the reasoning field.
    # Qwen3 on older Ollama puts the answer inside message.model_extra["reasoning"]
    # when thinking is not suppressed.  The three-stage JSON extractor below will
    # find the JSON anywhere inside that text.
    if not content:
        content = (msg.model_extra or {}).get("reasoning", "") or ""

    # Failure mode 3: JSON parse - try strict → fenced → bare.
    payload = _safe_json(content)
    if not payload:
        m = _FENCED_JSON.search(content)
        if m:
            payload = _safe_json(m.group(1))
    if not payload:
        m = _BARE_JSON.search(content)
        if m:
            payload = _safe_json(m.group(1))

    return Result(
        provider="ollama",
        label=normalise_label(payload.get("intent", "unknown")) if payload else "unknown",
        confidence=float(payload.get("confidence", 0.5)) if payload else 0.5,
        latency_ms=latency_ms,
        input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        raw={"content": content[:300]},
    )


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
_DISPATCH = {
    "openai":     openai_classify,
    "nano": nano_classify,
    "ollama":     ollama_classify,
}


def classify(provider: Provider, text: str) -> Result:
    """Single public entry point. App code only ever calls this."""
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    return fn(text)
