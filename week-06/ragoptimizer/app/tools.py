"""Tool definitions for the citation pipeline.

CitationRAG (W5) uses a single tool: `retrieve_chunks`. RAGOptimizer (W6)
keeps the same tool shape but changes what happens behind it - HyDE +
reranker + compressor - without changing the tool's JSON schema.

Tools follow the `{schema, impl}` pair pattern from Week 1 plus a tiny
dispatcher. No agent framework.
"""
from __future__ import annotations

from typing import Any, Callable

from app.schemas import Chunk

# --- Tool implementations live in other modules; we import lazily to avoid
# import cycles when the /eval/compare path loads only the retriever.


def _retrieve_chunks_impl(query: str, k: int = 5) -> list[dict[str, Any]]:
    from app.retriever import retrieve  # lazy import: see app/retriever.py
    chunks: list[Chunk] = retrieve(query=query, k=k)
    return [c.model_dump() for c in chunks]


TOOLS: dict[str, dict[str, Any]] = {
    "retrieve_chunks": {
        "schema": {
            "name": "retrieve_chunks",
            "description": (
                "Retrieve top-k document chunks for a user query, using "
                "HyDE + reranker + compressor when enabled."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's question, rewritten if multi-turn."},
                    "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                },
                "required": ["query"],
            },
        },
        "impl": _retrieve_chunks_impl,
    },
}


def execute_tool(name: str, args: dict[str, Any]) -> Any:
    """Tiny dispatcher - never raises, unknown tool returns error dict."""
    if name not in TOOLS:
        return {"success": False, "error": "unknown_tool", "tool": name}
    impl: Callable[..., Any] = TOOLS[name]["impl"]
    try:
        return impl(**args)
    except Exception as exc:
        return {"success": False, "error": str(exc), "tool": name}
