"""
Golden-set runner + aggregator.

  - load a small hand-labelled JSONL golden set
  - for each example, call each provider, time it, score it (correct/incorrect)
  - aggregate per-provider: accuracy, p50, p95, $/1K calls

Pricing is intentionally a small constant table you can edit when providers change.
"""
from __future__ import annotations
import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Iterable

from app.config import get_settings
from app.llm import classify
from app.schemas import BenchmarkRow, BenchmarkSummary, GoldenExample, Provider

logger = logging.getLogger("intentiq.eval")


# $ per 1M tokens - illustrative, update from provider pricing pages.
# OpenAI:   https://openai.com/pricing
# Ollama:   local inference - zero API cost (hardware cost not modelled)
_PRICING: dict[str, dict[str, float]] = {
    "openai":     {"input": 0.75, "output": 4.50},   # gpt-5.4-mini
    "nano": {"input": 0.20, "output": 1.25},   # gpt-5.4-nano-2026-03-17
    "ollama":     {"input": 0.00, "output": 0.00},   # qwen3:0.6b - local, zero API cost
}


def load_golden(path: str | Path) -> list[GoldenExample]:
    """
    Read *path* (a JSONL file) and return a list of GoldenExample objects.
    Blank lines are skipped.  Each non-blank line must be a valid JSON object
    matching the GoldenExample schema (id, input, expected, note?).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(GoldenExample(**json.loads(line)))
    return rows


def run_one(provider: Provider, example: GoldenExample) -> BenchmarkRow:
    """
    Run a single golden example through *provider* and return a scored BenchmarkRow.
    Captures latency, token counts, and whether the predicted label matches expected.
    Raises any provider exception - callers (run_benchmark, benchmark_stream) catch it.
    """
    res = classify(provider, example.input)
    return BenchmarkRow(
        example_id=example.id,
        provider=provider,
        predicted=res.label,
        expected=example.expected,
        correct=(res.label == example.expected),
        confidence=res.confidence,
        latency_ms=res.latency_ms,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
    )


def run_benchmark(
    golden: list[GoldenExample],
    providers: Iterable[Provider] = ("openai", "nano", "ollama"),
) -> list[BenchmarkRow]:
    """
    Run every example in *golden* through each provider in *providers*.

    Skips OpenAI-backed providers (openai, nano) if OPENAI_API_KEY is not set.
    Ollama always runs - it is local and needs no key.
    Per-example failures are logged as warnings and recorded as incorrect rows
    (latency 0, label 'unknown') so a single error doesn't abort the whole run.

    Returns a flat list of BenchmarkRow - one row per (provider, example) pair.
    """
    rows: list[BenchmarkRow] = []
    settings = get_settings()
    for provider in providers:
        # Skip OpenAI-backed providers if no key is configured.
        if provider in ("openai", "nano") and not settings.openai_api_key:
            continue
        # Ollama always runs - it's local and needs no key.
        # If Ollama isn't running the errors are logged row-by-row (see except below).
        for ex in golden:
            try:
                rows.append(run_one(provider, ex))
            except Exception as exc:  # noqa: BLE001
                # Log the real error so it appears in the uvicorn console,
                # then keep going - one failure shouldn't abort the whole run.
                logger.warning(
                    "provider=%s example_id=%s FAILED: %s", provider, ex.id, exc
                )
                rows.append(BenchmarkRow(
                    example_id=ex.id, provider=provider, predicted="unknown",
                    expected=ex.expected, correct=False, confidence=0.0,
                    latency_ms=0.0, input_tokens=0, output_tokens=0,
                ))
    return rows


def summarise(rows: list[BenchmarkRow]) -> list[BenchmarkSummary]:
    """
    Aggregate a flat list of BenchmarkRows into one BenchmarkSummary per provider.

    Computes per-provider:
      - accuracy   - fraction of correct predictions
      - p50_ms     - median latency (warm rows only, latency > 0)
      - p95_ms     - 95th-percentile latency
      - cost_per_1k_usd - estimated API cost per 1 000 calls using _PRICING table

    Returns summaries in the order providers appear in *rows*.
    """
    out: list[BenchmarkSummary] = []
    by_provider: dict[str, list[BenchmarkRow]] = {}
    for r in rows:
        by_provider.setdefault(r.provider, []).append(r)

    for provider, items in by_provider.items():
        latencies = [r.latency_ms for r in items if r.latency_ms > 0]

        p50 = statistics.median(latencies) if latencies else 0.0
        # statistics.quantiles requires >= 2 points; fall back to max for small sets.
        p95 = (
            statistics.quantiles(latencies, n=20)[-1]
            if len(latencies) >= 2 else (max(latencies) if latencies else 0.0)
        )

        n = len(items)
        accuracy    = sum(1 for r in items if r.correct) / n if n else 0.0
        in_tok      = sum(r.input_tokens  for r in items)
        out_tok     = sum(r.output_tokens for r in items)
        price       = _PRICING.get(provider, {"input": 0.0, "output": 0.0})
        cost_total  = (in_tok * price["input"] + out_tok * price["output"]) / 1_000_000
        cost_per_1k = (cost_total / n) * 1000 if n else 0.0

        out.append(BenchmarkSummary(
            provider=provider,    # type: ignore[arg-type]
            n=n,
            accuracy=accuracy,
            p50_ms=p50,
            p95_ms=p95,
            cost_per_1k_usd=cost_per_1k,
            cold_start_ms=None,
        ))
    return out


def write_csv(rows: list[BenchmarkRow], path: str | Path) -> None:
    """
    Write *rows* to a CSV file at *path*, overwriting any existing file.
    Columns are derived from BenchmarkRow's field names via model_dump().
    Does nothing if *rows* is empty.
    """
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].model_dump().keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.model_dump())
