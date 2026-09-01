"""Pydantic models for tool I/O and HTTP surface."""
from typing import Literal
from pydantic import BaseModel, Field


class LookupRequest(BaseModel):
    """Inbound HTTP body for POST /lookup."""
    message: str = Field(..., min_length=1, description="User's natural-language question.")
    provider: Literal["openai", "nano"] = Field(
        "openai",
        description=(
            "Which model to use. "
            "'openai' = gpt-5.4-mini-2026-03-17 (full reasoning). "
            "'nano'   = gpt-5.4-nano-2026-03-17 (small & fast)."
        ),
    )


class OrderLookupArgs(BaseModel):
    """Arguments the model must produce when calling lookup_order_status.

    The descriptions here are PROMPT CONTENT - they ship to the model
    inside the tool's parameters schema on every call.
    """
    order_id: str = Field(
        ...,
        description=(
            "The customer's order ID. Format ORD-NNNN, e.g. ORD-1042. "
            "Must be a string with the ORD- prefix. Do NOT pass an integer."
        ),
    )


class OrderStatus(BaseModel):
    """Typed result of the order lookup tool."""
    order_id: str
    state: Literal["pending", "shipped", "delivered", "cancelled"]
    eta: str
    last_update: str


class LookupResponse(BaseModel):
    """Outbound HTTP body for POST /lookup."""
    answer: str
    tool_call_count: int
    retry_count: int
    provider: str
    model: str
    latency_ms: float
