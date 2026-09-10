"""Typed settings loaded from .env via pydantic-settings.

No business code should touch os.environ directly. This is the single
source of truth for runtime configuration.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        extra="ignore", protected_namespaces=(),
    )

    hf_token: str = ""
    hf_hub_repo: str = "your-org/specialisttuner-qwen3-lora"
    base_model: str = "Qwen/Qwen3-0.6B"
    dataset_path: Path = Path("./data/specialisttuner.jsonl")
    output_dir: Path = Path("./out")
    log_dir: Path = Path("./logs")
    model_host: str = "0.0.0.0"
    model_port: int = 8000
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
