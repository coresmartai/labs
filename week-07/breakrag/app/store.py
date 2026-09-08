"""One Qdrant client for the whole process, in the two modes the course has
used since Week 4 plus the snapshot mode Project 14 needs.

  local   (default)  the embedded store, reading QDRANT_LOCAL_PATH. This is the
                     qdrant_local/ folder Week 4's KnowledgeVault wrote and
                     Week 5's CitationRAG read. The embedded store allows ONE
                     process at a time, so stop the Week 4, 5 or 6 server
                     before starting this one, and every module in this
                     process shares the single client below.
  server             a running Qdrant at QDRANT_URL (Docker or Qdrant Cloud),
                     with QDRANT_API_KEY for a hosted instance. The CI live job
                     uses this mode.

Switching modes is a .env change; no code path cares which one it got.
"""

from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger("breakrag.store")

_client = None


def get_client():
    """Return the process-wide Qdrant client, creating it on first use."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        s = get_settings()
        mode = s.qdrant_mode.lower()
        if mode == "snapshot":
            raise RuntimeError(
                "QDRANT_MODE=snapshot has no live store: the built-in /answer "
                "route is unavailable, and the leakage checks read "
                f"CORPUS_SNAPSHOT_PATH={s.corpus_snapshot_path}. Point "
                "SUT_BASE_URL at the service under test, or set QDRANT_MODE "
                "to local or server."
            )
        if mode == "local":
            from pathlib import Path

            if not Path(s.qdrant_local_path).is_dir():
                raise RuntimeError(
                    f"QDRANT_LOCAL_PATH={s.qdrant_local_path} is not a folder. Point it "
                    "at the qdrant_local/ folder your Week 4 ingestion wrote (the "
                    "default assumes the labs clone layout). Opening a wrong path would "
                    "silently create an empty store there."
                )
            _client = QdrantClient(path=s.qdrant_local_path)
            logger.info("Qdrant embedded store: %s (collection %s)", s.qdrant_local_path, s.qdrant_collection)
        else:
            _client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key or None)
            logger.info("Qdrant server: %s (collection %s)", s.qdrant_url, s.qdrant_collection)
    return _client
