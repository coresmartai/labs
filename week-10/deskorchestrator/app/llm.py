"""The provider seam. Everything above this is provider-agnostic."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import get_settings

log = logging.getLogger(__name__)

NETWORK_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@retry(retry=retry_if_exception_type(NETWORK_ERRORS),
       stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4),
       reraise=True)
def _call_provider(model: str, system: str, user: str, max_tokens: int) -> LLMResult:
    from openai import OpenAI  # imported late so the tests never need the package configured

    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key, timeout=s.request_timeout_seconds)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_completion_tokens=max_tokens,
    )
    usage = getattr(resp, "usage", None)
    return LLMResult(
        text=(resp.choices[0].message.content or "").strip(),
        model=model,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        elapsed_ms=int((time.time() - t0) * 1000),
    )


def call(model: str, system: str, user: str, max_tokens: int | None = None) -> LLMResult:
    """One call. Tests replace this function wholesale, so nothing below it runs offline."""
    s = get_settings()
    return _call_provider(model, system, user, max_tokens or s.max_completion_tokens)
