"""
Pydantic models that flow through the KnowledgeVault pipeline.

Why schema-first? Because the moment you let any stage emit untyped dicts,
your downstream stages start tolerating fields that don't exist and ignoring
fields that do. The pipeline becomes opaque. Pin the shape at every hop.

This file is intentionally the schema spine. If you change a field here, you
change it in exactly one place - the parser, chunker, embedder, indexer,
and retriever all import from this module.
"""

from __future__ import annotations
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ChunkType = Literal["prose", "figure-description", "table-row"]
Confidence = Literal["high", "medium", "low"]


# --------------------------------------------------------------
# Vision LLM structured output - see app/vision.py
# --------------------------------------------------------------

class FigureDescription(BaseModel):
    """Structured output from the vision LLM for a single figure crop."""

    type: str = Field(
        ...,
        description="Short tag - e.g. 'architecture diagram', 'sequence diagram', 'chart', 'screenshot'.",
    )
    summary: str = Field(
        ...,
        max_length=800,
        description="2-4 sentence description. Length cap enforced here, not in the tool schema (strict mode drops maxLength).",
    )
    key_elements: list[str] = Field(
        default_factory=list,
        max_length=13,
        description="Salient elements named in the figure (boxes, arrows, labels). Item cap enforced here, not in the tool schema.",
    )
    confidence: Confidence = Field(
        ...,
        description="High = trust this description. Low = consider skipping or re-describing.",
    )


# --------------------------------------------------------------
# Parser output - one item per text block / figure / table region
# --------------------------------------------------------------

class ParsedBlock(BaseModel):
    """A single parsed region from a PDF page."""

    document_id: str
    source_url: str
    page_number: int
    block_type: Literal["text", "figure", "table"]
    text: str | None = None
    image_path: str | None = None
    table_rows: list[list[str]] | None = None
    bbox: tuple[float, float, float, float]
    section_heading: str | None = None


# --------------------------------------------------------------
# Chunker output - the thing we embed + index
# --------------------------------------------------------------

class Chunk(BaseModel):
    """A retrievable chunk with metadata."""

    chunk_id: str
    parent_id: str | None = None
    document_id: str
    source_url: str
    section_heading: str | None = None
    page_number: int
    chunk_index: int
    created_at: datetime
    chunk_type: ChunkType
    text: str
    image_path: str | None = None
    figure_ids: list[str] = Field(default_factory=list)


# --------------------------------------------------------------
# Retrieval API
# --------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    document_id: str | None = None
    # Match on the small child, then widen to its 1,500-token parent: when
    # True, every prose hit also carries parent_text, the stitched text of all
    # children that share its parent_id. This is what the hierarchy is for.
    widen: bool = False

class RetrievedChunk(BaseModel):
    chunk_id: str
    # Which PDF the hit came from. Always set; without it a multi-document
    # corpus cannot tell the caller where an answer lives, and the Week 4
    # project's benchmark scores hits by document.
    document_id: str
    chunk_type: ChunkType
    text: str
    page_number: int
    section_heading: str | None = None
    image_url: str | None = None
    score: float
    # Hierarchy. parent_id is set on prose chunks. parent_text is filled only
    # when the request asked to widen; it is the parent span reassembled from
    # its children with the 50-token overlaps removed.
    parent_id: str | None = None
    parent_text: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    fusion: Literal["rrf", "weighted", "rerank"] = "rrf"
