"""GuardianAI - a hardening layer for a citation-based RAG service.

Adds PII scrubbing (Presidio), prompt-injection defence, RBAC + ACL, and an
append-only audit log on top of the CitationRAG retrieval service.

The package is deliberately small: every module reads top to bottom, and every
behaviour is exercised by the eval set in tests/.
"""
