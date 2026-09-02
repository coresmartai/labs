"""
One place that knows how to reach Qdrant.

Two modes, chosen by QDRANT_MODE in .env:

  local   (default)  Qdrant runs embedded in this Python process and persists
                     to QDRANT_LOCAL_PATH on disk. Nothing to install, nothing
                     to sign up for. One rule: only one process may hold the
                     local store at a time, so run `python -m app.ingest` with
                     the server stopped, or ingest through POST /ingest while
                     it is running. Never both at once.

  server             A Qdrant server at QDRANT_URL: `docker run qdrant/qdrant`
                     on your laptop, or a Qdrant Cloud cluster with
                     QDRANT_API_KEY set. Many processes may connect.

Every module that needs a client calls get_client(). Nobody constructs
QdrantClient anywhere else, so switching modes is one line in .env.
"""

from qdrant_client import QdrantClient

from app.config import get_settings


def get_client() -> QdrantClient:
    """Return a Qdrant client for the configured mode."""
    settings = get_settings()
    if settings.qdrant_mode.lower() == "local":
        return QdrantClient(path=settings.qdrant_local_path)
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
