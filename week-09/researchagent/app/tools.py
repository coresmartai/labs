"""Three tools, described to the model as JSON Schema.

Same shape as OpsAssist: a dict from name to {input_model, output_model, impl,
description}, and a dispatcher that never raises. What changed is the catalogue,
which is the point of the week.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.retrieval import CorpusIndex

logger = logging.getLogger(__name__)


class SearchCorpusInput(BaseModel):
    query: str = Field(description="Keywords to search the corpus for. Not a full sentence.")
    k: int = Field(default=5, ge=1, le=8)


class ReadChunkInput(BaseModel):
    chunk_uid: str = Field(description="A chunk_uid returned by search_corpus, e.g. rfc9989:0042")


class ListDocumentsInput(BaseModel):
    pass


def build_tools(index: CorpusIndex) -> dict[str, dict[str, Any]]:
    """The catalogue is built against an index, so the tools close over the corpus."""

    def search_corpus(args: SearchCorpusInput) -> dict:
        hits = index.search(args.query, k=args.k)
        return {
            "success": True,
            "query": args.query,
            "hits": [
                {"rank": h.rank, "chunk_uid": h.chunk.chunk_uid, "rfc": h.chunk.rfc,
                 "section": h.chunk.section, "page": h.chunk.page,
                 "preview": h.chunk.text[:300]}
                for h in hits
            ],
        }

    def read_chunk(args: ReadChunkInput) -> dict:
        chunk = index.get(args.chunk_uid)
        if chunk is None:
            return {"success": False, "error": "no_such_chunk", "chunk_uid": args.chunk_uid,
                    "terminal": True}
        return {"success": True, "chunk_uid": chunk.chunk_uid, "rfc": chunk.rfc,
                "section": chunk.section, "page": chunk.page, "text": chunk.text}

    def list_documents(args: ListDocumentsInput) -> dict:
        return {"success": True, "documents": index.documents,
                "note": "This is the whole corpus. Nothing outside it is available to you."}

    return {
        "search_corpus": {
            "input_model": SearchCorpusInput,
            "impl": search_corpus,
            "description": (
                "Search the corpus for passages matching keywords. Returns ranked previews "
                "with a chunk_uid for each. Call this first for any question, and call it "
                "again with different keywords when the first results do not settle it. "
                "The rank is position in THIS result set only and means nothing between calls."
            ),
        },
        "read_chunk": {
            "input_model": ReadChunkInput,
            "impl": read_chunk,
            "description": (
                "Return the full text of one chunk by its chunk_uid. Use this before making "
                "a claim from a passage whose preview was truncated. A quote you have not "
                "read in full is a quote you cannot cite."
            ),
        },
        "list_documents": {
            "input_model": ListDocumentsInput,
            "impl": list_documents,
            "description": (
                "List the documents in the corpus. Use this when you are unsure whether a "
                "question is answerable from the corpus at all. If the subject is not in "
                "these documents, say so and stop rather than searching repeatedly."
            ),
        },
    }


def execute_tool(tools: dict[str, dict[str, Any]], name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch. Always returns a dict, never raises. Same contract as OpsAssist."""
    if name not in tools:
        logger.warning("Unknown tool: %s", name)
        return {"success": False, "error": "unknown_tool", "tool": name, "terminal": True}
    spec = tools[name]
    try:
        parsed = spec["input_model"](**args)
        return spec["impl"](parsed)
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return {"success": False, "error": str(exc), "tool": name, "terminal": False}


def catalog_for_model(tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {
            "name": name,
            "description": spec["description"],
            "parameters": spec["input_model"].model_json_schema(),
        }}
        for name, spec in tools.items()
    ]
