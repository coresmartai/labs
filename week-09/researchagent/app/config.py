"""Typed settings, loaded once.

All configuration goes through this module. No `os.environ` calls anywhere
else in the codebase. Missing required env vars fail loudly at import time.
"""
import logging
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------
# Price table - US dollars per MILLION tokens, list price as published on
# each model's own page, checked 10 Sep 2026. Note the snapshot dates are
# NOT uniform across the family: mini and nano are 2026-03-17, the base
# model is 2026-03-05. Copying one date across all three is the mistake
# this table exists to not make. Cached input is exactly 10% of the input
# rate for all three, so it is derived rather than typed out (one number
# to get wrong instead of two).
# ---------------------------------------------------------------------
CACHED_INPUT_DISCOUNT = 0.10

MODEL_PRICES: dict[str, tuple[float, float]] = {
    # model                        (input, output)
    "gpt-5.4-nano-2026-03-17":     (0.20, 1.25),
    "gpt-5.4-mini-2026-03-17":     (0.75, 4.50),
    "gpt-5.4-2026-03-05":          (2.50, 15.00),
}

log = logging.getLogger(__name__)

# If someone pins a model we have no price for, fall back to the most
# expensive row rather than silently under-counting the spend. A ceiling
# that errs expensive still protects you; one that errs cheap is not a
# ceiling at all. But silence is the defect, not the fallback, so this
# path logs every time it fires.
_FALLBACK_MODEL = "gpt-5.4-2026-03-05"
_FALLBACK_PRICE = MODEL_PRICES[_FALLBACK_MODEL]


class Settings(BaseSettings):
    """Single source of truth for runtime configuration.

    Loaded from a `.env` file in the project root. See `.env.example`
    for the complete list of required variables.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=())

    # Model selection - pinned to a specific version, never `-latest`.
    openai_model: str = "gpt-5.4-mini-2026-03-17"
    openai_api_key: str

    # Agent loop safety devices. The cap is per QUESTION, not per run: three
    # questions at ten iterations each is a worst case of thirty model calls.
    max_iters: int = 10
    max_wall_clock_seconds: int = 120
    request_timeout_seconds: int = 60
    # Per-completion output cap (distinct from the whole-run token budget below).
    max_completion_tokens: int = 4096

    # The whole-run token budget, per question.
    token_budget: int = 16000

    # The dollar-cost stop. Same pattern as the token budget: multiply tokens
    # by price, stop at the ceiling. Off by default - set MAX_COST_USD to a
    # float (e.g. 0.05) to arm it.
    max_cost_usd: float | None = None

    # Prices for the pinned model, in dollars per million tokens. Left unset,
    # they are looked up in MODEL_PRICES from `openai_model` - so the rates
    # follow the model you pin instead of drifting away from it. Override any
    # of the three explicitly if the list price moves.
    price_per_1m_input_usd: float | None = None
    price_per_1m_output_usd: float | None = None
    price_per_1m_cached_input_usd: float | None = None

    # ---------------------------------------------------------------
    # YOUR STOP CONDITION GOES HERE.
    #
    # The four guards inherited from OpsAssist - the iteration cap, the wall
    # clock, the token budget and the dollar ceiling - all answer "has this run
    # cost too much?". None of them answers "is this research finished?",
    # because that is a coverage question and OpsAssist never had one.
    #
    # Add whatever settings your condition needs. Two examples, neither of
    # which is the answer:
    #   sufficient_evidence_entries: int = 4
    #   no_new_evidence_rounds: int = 2
    #
    # Whatever you choose, you defend it in the memo. An inherited iteration
    # cap is not a research stop condition, and saying so is worth two marks.
    # ---------------------------------------------------------------

    log_level: str = "INFO"

    @model_validator(mode="after")
    def _resolve_prices(self) -> "Settings":
        """Fill any unset price from MODEL_PRICES, keyed on the pinned model."""
        if self.openai_model in MODEL_PRICES:
            table_in, table_out = MODEL_PRICES[self.openai_model]
        else:
            table_in, table_out = _FALLBACK_PRICE
            log.warning(
                "No price row for model %r. Falling back to %r at $%.2f in / $%.2f out "
                "per million tokens, which over-counts rather than under-counts. "
                "Add a row to MODEL_PRICES, or set PRICE_PER_1M_INPUT_USD and "
                "PRICE_PER_1M_OUTPUT_USD explicitly.",
                self.openai_model, _FALLBACK_MODEL, table_in, table_out,
            )
        if self.price_per_1m_input_usd is None:
            self.price_per_1m_input_usd = table_in
        if self.price_per_1m_output_usd is None:
            self.price_per_1m_output_usd = table_out
        if self.price_per_1m_cached_input_usd is None:
            # Cached input is exactly 10% of input. Always.
            self.price_per_1m_cached_input_usd = round(
                self.price_per_1m_input_usd * CACHED_INPUT_DISCOUNT, 6
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the (effectively singleton) settings object."""
    return Settings()
