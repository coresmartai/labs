"""Typed settings - loaded once via lru_cache, never read via os.environ in business code."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pinned configuration for the RAGOptimizer service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=())

    # --- OpenAI / model ---
    openai_api_key: str = ""
    model_id: str = "gpt-5.4-mini-2026-03-17"
    hyde_model: str = "gpt-5.4-nano-2026-03-17"

    # --- Retrieval / Qdrant (matches Week 4 KnowledgeVault naming exactly) ---
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant Cloud URL or local address")
    qdrant_api_key: str = Field(default="", description="Qdrant Cloud API key; leave empty for local")
    qdrant_collection: str = Field(default="knowledgevault", description="Collection name - must match Week 4 index")
    openai_embed_model: str = Field(default="text-embedding-3-large", description="3072-dim model used to index Week 4 data")
    vector_top_k_wide: int = Field(30, ge=5, le=100)
    vector_top_k_narrow: int = Field(5, ge=1, le=20)

    # --- Threshold gate (BOTH columns in /eval/compare, and /ask) ---
    # Calibrated for text-embedding-3-large cosine scores and carried over from
    # Week 5 unchanged, so the baseline refuses on exactly the queries W5 refused.
    # The gate reads dense cosines (top1/spread), never the RRF score.
    #
    # The full W6 pipeline uses these too, at the SAME values - it just evaluates
    # the rule against both of its probes (raw query OR HyDE) instead of one:
    #     refuse unless (raw_ok or hyde_ok)
    # Gating only the baseline would confound the comparison - the 'after' column
    # would win refusal rows by not having a gate rather than by retrieving
    # better. Symmetric gate in, retrieval quality out.
    similarity_threshold: float = Field(0.55, ge=0.0, le=1.0, description="Threshold gate: minimum top-1 dense cosine (both pipelines)")
    spread_delta: float = Field(0.08, ge=0.0, le=1.0, description="Threshold gate: minimum top1-top3 dense cosine spread (both pipelines)")

    # --- HyDE ---
    hyde_enabled: bool = True
    hyde_prompt_voice: str = "academic_researcher"

    # --- Reranker ---
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = Field(30, ge=1, le=128)
    reranker_cpu_threads: int = Field(4, ge=1, le=32)

    # --- Compressor ---
    compressor_enabled: bool = True
    compressor_keep_fraction: float = Field(0.80, ge=0.10, le=1.0)

    # --- Service SLA ---
    p95_latency_budget_ms: int = Field(2000, ge=500, le=10000)

    # --- Benchmarking ---
    golden_dataset_path: str = "./data/golden_dataset.json"
    bootstrap_resamples: int = Field(10000, ge=100, le=100000)

    # Logging
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
