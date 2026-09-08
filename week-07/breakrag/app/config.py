"""Typed runtime settings loaded once from environment variables.

We use pydantic-settings BaseSettings + @lru_cache so the environment is
parsed exactly once per process. Route handlers and test code should only
ever read settings via `get_settings()`, never via `os.environ` directly.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- system under test --------------------------------------------------
    sut_base_url: str = "http://localhost:8000"
    sut_timeout_seconds: int = 30

    # --- judge provider ------------------------------------------------------
    openai_api_key: str | None = None

    # --- pinned model versions (all dated - no "latest" tags) ----------------
    # Judge PRIMARY is the nano tier (fulfils W5's "nano as judge" promise).
    # Judge SECONDARY is the mini tier. Caveat: the secondary judge shares the
    # SUT's model; the Δ>1 disagreement flag and the independent primary judge
    # cover the self-grading bias risk.
    openai_model: str = "gpt-5.4-mini-2026-03-17"        # judge secondary
    openai_model_nano: str = "gpt-5.4-nano-2026-03-17"   # judge primary
    sut_model: str = "gpt-5.4-mini-2026-03-17"           # answer generation (SUT)

    # --- harness thresholds (V1 statistical discipline) ----------------------
    # These four are the CI floors. They are asserted in tests/test_eval_harness.py
    # and they are the four numbers the concept video sells on camera.
    adversarial_pass_rate_floor: float = Field(default=0.70, ge=0.0, le=1.0)
    judge_agreement_alpha_floor: float = Field(default=0.60, ge=0.0, le=1.0)
    min_samples: int = Field(default=50, ge=1)
    bootstrap_resamples: int = Field(default=5000, ge=200)

    # A case whose two judges are more than this far apart on the 0-5 rubric is
    # ambiguous, not simply good or bad. Reported (judge_disagreement_rate) and
    # flagged for human review - deliberately NOT a fifth CI floor.
    judge_disagreement_delta: float = Field(default=1.0, ge=0.0, le=5.0)

    # Embedding-similarity leakage check: flag a seed whose nearest corpus
    # neighbour sits above this percentile of the corpus's own neighbour
    # distribution. Calibrated against your corpus - never a hard 0.95 cosine.
    embedding_leakage_percentile: float = Field(default=99.0, ge=50.0, le=100.0)
    # Upper bound on how many indexed passages feed the embedding check's
    # corpus-vs-corpus baseline (the scan is quadratic). Seeds are still
    # compared against every passage. 0 = no cap.
    leakage_corpus_cap: int = Field(default=600, ge=0)

    # --- adversarial generator ----------------------------------------------
    # 7 = 1 SEED + 2 TYPO + 1 each of JAILBREAK / MULTI_HOP / CONFLICT / HOSTILE
    # → 10 seeds × 7 = 70 cases for the classroom run.
    red_team_cases_per_seed: int = Field(default=7, ge=1)
    seed_set_path: str = "app/seeds/golden_set.json"

    # --- retriever (KnowledgeVault / CitationRAG Qdrant collection) --
    # The same store Week 4 wrote and Week 5 read. Two modes, as in Week 5:
    #   local  (default) = the embedded store at QDRANT_LOCAL_PATH. One process
    #                      at a time: stop the earlier weeks' servers first.
    #   server           = a running Qdrant at QDRANT_URL (Docker or Cloud).
    #   snapshot         = no live store. The leakage checks read a corpus
    #                      snapshot written by scripts/snapshot_corpus.py, so
    #                      the harness never opens the embedded folder while
    #                      the service under test holds it (Project 14).
    qdrant_mode: str = "local"
    corpus_snapshot_path: str = "runs/corpus_snapshot.json"
    qdrant_local_path: str = "../../week-04/knowledgevault/qdrant_local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "knowledgevault"

    # Embedding model - must match what was used to index the collection.
    embed_model: str = "text-embedding-3-large"

    # Retrieval thresholds (calibrated for text-embedding-3-large cosine scores).
    retrieval_top_k: int = Field(default=5, ge=1)
    similarity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    spread_delta: float = Field(default=0.08, ge=0.0, le=1.0)

    # --- logging --------------------------------------------------------------
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cache once; the env doesn't change inside a process."""
    return Settings()
