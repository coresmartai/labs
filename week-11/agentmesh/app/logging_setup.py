"""One place that wires up logging - console, plus a rotating file on disk.

Every module already does `logging.getLogger(__name__)` and logs to it. Those
records propagate to the ROOT logger, so attaching one rotating file handler to
the root is all it takes for the whole app - main, task_store, jwt_verify, llm,
mcp_server, uvicorn's own access logs - to auto-persist. No per-module wiring, and
no button to press: a session leaves `logs/agentmesh.log` behind on its own.

Called once at startup from `app/main.py`. Idempotent, because uvicorn's reloader
imports the app twice and we must not stack a second file handler each time.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from app.config import Settings

_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
_MARK = "_agentmesh_file"   # tags our handler so a reload does not add a second one


def setup_logging(settings: Settings) -> Path | None:
    """Configure the root logger. Returns the log-file path, or None if disabled."""
    root = logging.getLogger()
    root.setLevel(settings.log_level)

    # Console handler - the same console output a plain basicConfig call produces.
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)

    if not settings.log_to_file:
        return None

    # Already attached (reloader re-import)? Do nothing, return the existing path.
    for h in root.handlers:
        if getattr(h, _MARK, False):
            return Path(getattr(h, "baseFilename", settings.log_file))

    path = Path(settings.log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FORMAT))
    setattr(file_handler, _MARK, True)
    root.addHandler(file_handler)
    return path.resolve()
