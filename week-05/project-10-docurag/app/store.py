"""
One place that knows how to reach Qdrant. Same pattern as Week 4 KnowledgeVault.

Two modes, chosen by QDRANT_MODE in .env:

  local   (default)  Qdrant runs embedded in this Python process and reads the
                     store at QDRANT_LOCAL_PATH on disk. Nothing to install,
                     nothing to sign up for. Point it at the qdrant_local/
                     folder your Week 4 KnowledgeVault ingestion created (or
                     copy that folder into this project). One rule: only one
                     process may hold the local store at a time, so stop the
                     Week 4 server before starting CitationRAG.

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
