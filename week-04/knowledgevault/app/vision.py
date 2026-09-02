"""
Thin alias over app.llm.describe_figure with a confidence filter.

Why a separate module?
  The chunker (and anything else that needs figure descriptions) imports from
  `app.vision`, not directly from `app.llm`. This keeps the provider boundary
  in one place: swapping from gpt-5.4-mini to a different vision model means
  touching only `app.llm` - all callers continue to import `app.vision` unchanged.

What does drop_low_confidence protect against?
  When the vision model cannot clearly read a chart (e.g. blurry scan, very
  small figure, or text rendered as pixels), it is instructed to mark the
  description as "low" confidence rather than guess at numbers. Indexing those
  descriptions would pollute the retrieval corpus with hallucinated values -
  a user asking "what is the P99 latency?" could get a fabricated answer.
  Setting drop_low_confidence=True (the default) silently drops those blocks
  at ingestion time so they never reach the vector store.

Public surface:
  describe_figure(
      image_bytes: bytes,
      media_type: str = "image/png",
      *,
      drop_low_confidence: bool = True,
  ) -> Optional[FigureDescription]
"""

from __future__ import annotations

import logging
from typing import Optional

from app.llm import describe_figure as _describe
from app.schemas import FigureDescription

logger = logging.getLogger(__name__)


def describe_figure(
    image_bytes: bytes,
    media_type: str = "image/png",
    *,
    drop_low_confidence: bool = True,
) -> Optional[FigureDescription]:
    """
    Describe a figure image via the vision LLM, optionally discarding
    low-confidence results before they reach the index.

    Args:
        image_bytes: Raw bytes of the figure image.
        media_type: MIME type of the image, e.g. "image/png" or "image/jpeg".
        drop_low_confidence: When True (default), returns None for any
            description the model marked as low-confidence. Prevents
            hallucinated chart values from entering the vector store.

    Returns:
        A FigureDescription, or None if the result was dropped.
    """
    desc: FigureDescription = _describe(image_bytes, media_type)

    if drop_low_confidence and desc.confidence == "low":
        logger.warning(
            "describe_figure: dropping low-confidence description (type=%s)",
            desc.type,
        )
        return None

    return desc
