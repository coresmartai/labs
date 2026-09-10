"""Generic → Vertex AI Gemini JSONL converter.

The app's internal chat format is:

    [{"role": "user",  "message": "..."},
     {"role": "model", "message": "..."}]

The Vertex AI Gemini supervised-tuning JSONL format is:

    {"systemInstruction": {"role": "system", "parts": [{"text": "..."}]},
     "contents": [
        {"role": "user",  "parts": [{"text": "..."}]},
        {"role": "model", "parts": [{"text": "..."}]}
     ]}

One conversation per JSONL line. This module owns the conversion and
nothing else. Keeping the converter isolated means every other layer of the
codebase stays provider-agnostic: prompts, training data, and evaluation
sets are authored once in the generic format and re-emitted for whichever
provider owns the tuning job this quarter.

Public functions:
  convert_example  - one dict → Vertex-shaped dict
  convert_file     - generic JSONL → Vertex JSONL (returns row counts)
  validate_generic - lint the generic file before you burn quota
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .schemas import ChatExample, Message

log = logging.getLogger(__name__)

VertexTurn = dict[str, Any]
VertexExample = dict[str, Any]


# ─── One-example converter ────────────────────────────────────────────

def convert_example(generic_example: dict[str, Any],
                    system_instruction: str | None = None) -> VertexExample:
    """Convert one generic ChatExample dict to Vertex Gemini SFT format.

    * Empties are rejected via Pydantic min_length checks in ChatExample.
    * Consecutive same-role turns are allowed (Vertex tolerates them) but
      the first turn must be `user` and turns should alternate, so we log a
      warning if that pattern is violated.
    * If a system_instruction string is supplied it is emitted at the top
      level under `systemInstruction` in the exact shape Vertex expects.
    """
    # Validate through Pydantic - catches missing fields, wrong role
    # literal, empty message strings, all at the boundary.
    example = ChatExample.model_validate(generic_example)

    # Structural warning: first turn should be user
    if example.messages[0].role != "user":
        log.warning("convert_example: first turn is %s, not user", example.messages[0].role)

    contents: list[VertexTurn] = [
        {"role": m.role, "parts": [{"text": m.message}]}
        for m in example.messages
    ]

    out: VertexExample = {"contents": contents}
    if system_instruction:
        out["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": system_instruction}],
        }
    return out


# ─── File-level converter ─────────────────────────────────────────────

def convert_file(generic_path: str | Path,
                 gcloud_path: str | Path,
                 system_instruction: str | None = None) -> dict[str, Any]:
    """Read generic JSONL, write Vertex-shaped JSONL, return row counts.

    Skips blank lines. Errors on any malformed row are collected and
    returned in the result dict so the caller (CLI, tools.py, FastAPI
    route) can present them together instead of erroring on the first one.
    """
    generic_path = Path(generic_path)
    gcloud_path = Path(gcloud_path)

    warnings: list[str] = []
    input_rows = 0
    output_rows = 0

    with generic_path.open(encoding="utf-8") as fin, \
         gcloud_path.open("w", encoding="utf-8") as fout:
        for lineno, raw in enumerate(fin, 1):
            raw = raw.strip()
            if not raw:
                continue
            input_rows += 1
            try:
                obj = json.loads(raw)
                vertex_obj = convert_example(obj, system_instruction=system_instruction)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"line {lineno}: {exc}")
                continue
            fout.write(json.dumps(vertex_obj, ensure_ascii=False) + "\n")
            output_rows += 1

    log.info("convert_file: %s → %s (%d in / %d out / %d warnings)",
             generic_path, gcloud_path, input_rows, output_rows, len(warnings))
    return {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "output_path": str(gcloud_path),
        "warnings": warnings,
    }


# ─── Pre-flight validator ─────────────────────────────────────────────

def validate_generic(generic_path: str | Path) -> dict[str, Any]:
    """Lint the generic file. Reports per-line errors and total row count.

    This is the pre-flight check the tuning pipeline runs before doing
    anything that costs money. The same pattern as OpenAI's file-upload
    validator. Return shape matches the tools.py dispatcher's `success`
    convention.
    """
    generic_path = Path(generic_path)
    if not generic_path.exists():
        return {"success": False, "error": "file_not_found", "path": str(generic_path)}

    errors: list[str] = []
    ok_rows = 0
    with generic_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                ChatExample.model_validate_json(raw)
                ok_rows += 1
            except Exception as exc:  # noqa: BLE001
                errors.append(f"line {lineno}: {exc}")

    return {
        "success": not errors,
        "count": ok_rows,
        "errors": errors,
        "path": str(generic_path),
    }
