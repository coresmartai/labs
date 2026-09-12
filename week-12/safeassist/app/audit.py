"""Append-only audit log writer.

One JSONL line per event. The file is opened with `O_APPEND` semantics
so concurrent writers do not interleave inside a line. The user ID is
hashed with a salt held in settings - production puts that salt in a
separate vault, never co-located with this file.

The single most common bug at this seam is logging a raw value into
`detail`. The test suite includes a grep over the audit file for any
known PII string from the test inputs; if a string leaks in, the test
fails. Treat the audit log itself as a piece of data that needs the
same evaluation discipline as the model output.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import cast

from app.config import get_settings
from app.schemas import AuditEvent

logger = logging.getLogger(__name__)


def hash_user_id(user_id: str) -> str:
    """Salted SHA-256. Salt lives in settings/vault, never in the log file."""
    settings = get_settings()
    h = hashlib.sha256()
    h.update(settings.user_id_hash_salt.encode("utf-8"))
    h.update(b":")
    h.update(user_id.encode("utf-8"))
    return h.hexdigest()


def write_event(
    *,
    request_id: str,
    user_id: str,
    event_type: str,
    detail: dict,
    action_taken: str,
) -> None:
    """Append one event to the audit log.

    Intentionally synchronous and unbuffered at the application layer.
    Loss aversion beats throughput here - a missing audit line is
    worse than a slow request.
    """
    settings = get_settings()
    event = AuditEvent(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        request_id=request_id,
        user_id_hash=hash_user_id(user_id),
        event_type=cast(str, event_type),  # narrow at the call site
        detail=detail,
        action_taken=action_taken,
    )
    line = event.model_dump_json() + "\n"
    fd = os.open(settings.audit_log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
