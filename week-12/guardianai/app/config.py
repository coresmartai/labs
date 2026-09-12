"""Typed Settings.

All env-var reads live here. Business code calls `get_settings()` and never
reaches into `os.environ`.

Everything this app talks to is real: a real OpenAI embedding model, a real
pinned chat model, a real Qdrant service (local Docker or a free Qdrant Cloud
cluster), and a real Presidio pipeline. There is no demo/offline switch in the
application code. The test suite gets its determinism by monkeypatching the two
network seams (`app.embedder` and `app.llm`), which is where a fake belongs.

`openai_api_key` has no default, so a missing key is a loud startup failure
before the first request rather than a 401 mid-run.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Required - the process exits loud on a missing key.
    openai_api_key: str
    openai_model: str = "gpt-5.4-mini-2026-03-17"
    # 3072-dim. Changing it changes the vector width, which requires a re-index:
    # `python -m app.ingest --reset`.
    openai_embed_model: str = "text-embedding-3-large"

    retrieval_top_k: int = Field(5, ge=1, le=50)
    # Minimum cosine score for a chunk to be returned at all, calibrated for
    # text-embedding-3-large. This is what keeps the ACL behaviour honest:
    # without a threshold Qdrant back-fills the result list with weak matches to
    # reach `limit`.
    similarity_threshold: float = Field(0.55, ge=0.0, le=1.0)

    spacy_model: str = "en_core_web_lg"

    # Crash the boot if Presidio is not actually operational. Turning this off
    # reproduces the silent-scrubber failure (a guardrail that fails quietly).
    startup_check_enabled: bool = True

    # Defence toggles - each turns one defence off so the attack it stops can be
    # observed landing, then back on. All default ON.
    injection_classifier_enabled: bool = True
    retrieval_sanitiser_enabled: bool = True
    chunk_scrub_enabled: bool = True

    presidio_redact_threshold: float = 0.65
    presidio_log_only_threshold: float = 0.35
    # Presidio's default operator is `replace`, substituting <ENTITY_TYPE>. We
    # also replace, but with one fixed <REDACTED> token, so every redaction reads
    # the same and the entity type lives in the audit log instead of the answer.
    # `redact` would DELETE the span, leaving a gap you cannot tell from a typo.
    presidio_operator: str = "replace"
    output_scrubber_max_buffer: int = 280
    request_timeout_seconds: int = 60

    audit_log_path: str = "./audit.jsonl"
    user_id_hash_salt: str = "change-me-32-hex-chars"

    # Qdrant - local Docker by default, or point at a free Qdrant Cloud
    # cluster (https://cloud.qdrant.io).
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "citation_rag"
    # qdrant-client passes no timeout by default, so httpx applies its own 5s.
    # A 3072-dim vector is ~43 KB of JSON, so even a modest batch approaches a
    # megabyte - over the public internet to a free-tier cluster that does not
    # fit in 5s, and the write fails with "The write operation timed out".
    qdrant_timeout_seconds: int = 60

    # Ingestion API limits (POST /ingest/file, POST /ingest/text).
    max_upload_bytes: int = 2_000_000
    chunk_chars: int = 400

    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached. The settings object is immutable for the process lifetime."""
    return Settings()
