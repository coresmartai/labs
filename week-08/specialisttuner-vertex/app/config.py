"""Typed settings for the Vertex AI Gemini supervised-tuning stack.

Every environment variable in the system flows through here. Business code
never touches os.environ directly. Cached once per process via @lru_cache.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        extra="ignore", protected_namespaces=(),
    )

    # ── GCP project + region ──────────────────────────────────────────
    gcp_project: str = "your-gcp-project-id"
    gcp_location: str = "us-central1"

    # ── Model pin (dated identifier, never use *-latest in production) ──
    source_model: str = "gemini-3.1-flash-lite-001"

    # ── GCS bucket for training artefacts ──────────────────────────────
    gcs_bucket: str = "gs://your-tuning-bucket"
    gcs_train_prefix: str = "aiayn/train"
    gcs_eval_prefix: str = "aiayn/eval"

    # ── Local paths ────────────────────────────────────────────────────
    generic_dataset_path: Path = Path("./data/aiayn_generic.jsonl")
    gcloud_dataset_path: Path = Path("./data/aiayn_gcloud.jsonl")
    eval_dataset_path: Path = Path("./data/eval_aiayn.jsonl")

    # ── Tuning hyperparameters ─────────────────────────────────────────
    tuning_epochs: int = 4
    tuning_adapter_size: int = 4
    learning_rate_multiplier: float = 1.0
    tuned_model_display_name: str = "specialisttuner-aiayn"

    # ── Serving ────────────────────────────────────────────────────────
    model_host: str = "0.0.0.0"
    model_port: int = 8000
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Single source of truth for settings, read from .env once."""
    return Settings()
