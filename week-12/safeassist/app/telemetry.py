"""The fifth egress path: your own operational logging.

The audit log is value-free by design and is covered. Ordinary telemetry is
not, and it is an egress path like any other. It usually has longer retention
than the database, wider access than the database, and no scrubber in front of
it, which makes it the quietest way for a redacted value to leave a service
that believes it redacts.

The rule this module enforces is the same rule the input scrubber follows:
**scrub before the write, not on the read.** A telemetry line is written once
and read by systems you do not control.

Two failure modes this is built to avoid.

**Scrubbing the message and forgetting the fields.** Structured logging puts
the interesting values in keyword arguments, not in the message string, and a
scrubber applied only to the message is a scrubber applied to the least likely
place for a value to be. `emit()` scrubs every string value in the payload,
recursively, as well as the message.

**Scrubbing on the way out of the logger.** A handler-level filter runs after
the record exists, which means the value has already been formatted, buffered,
and in some configurations already shipped. The scrub belongs at the call site.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.presidio_layer import scrub_telemetry


def _clean(value: Any) -> Any:
    """Recursively scrub every string in a payload.

    Numbers, booleans and None pass through untouched: Presidio has nothing to
    say about them and running the analyzer over an integer is pure cost.
    """
    if isinstance(value, str):
        return scrub_telemetry(value).cleaned_text
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def emit(event: str, message: str, **fields: Any) -> dict[str, Any]:
    """Write one scrubbed telemetry line. Returns the line for testing.

    Every string in `message` and in `fields` is scrubbed before the write.
    """
    settings = get_settings()
    line = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "event": event,
        "message": scrub_telemetry(message).cleaned_text,
        "fields": _clean(fields),
    }
    if not settings.telemetry_enabled:
        return line
    payload = json.dumps(line, separators=(",", ":")) + "\n"
    fd = os.open(settings.telemetry_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)
    return line
