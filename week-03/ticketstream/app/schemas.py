"""TicketStream Pydantic models - discriminated intent union, Literal priority."""
from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Intent variants (discriminated union, V2 pattern)
# ---------------------------------------------------------------------------

class OrderIntent(BaseModel):
    type: Literal["order"] = "order"
    order_id: str = Field(
        ...,
        description="The customer's order ID, format ORD-NNNN.",
    )


class RefundIntent(BaseModel):
    type: Literal["refund"] = "refund"
    order_id: str = Field(..., description="Order ID being refunded.")
    reason: str = Field(..., min_length=3, description="Refund reason.")


class BillingIntent(BaseModel):
    type: Literal["billing"] = "billing"
    invoice_id: str = Field(..., description="Invoice ID under question.")


IntentType = Annotated[
    Union[OrderIntent, RefundIntent, BillingIntent],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Attachment (nested model)
# ---------------------------------------------------------------------------

class Attachment(BaseModel):
    filename: str
    url: str


# ---------------------------------------------------------------------------
# TicketSchema (top-level)
# ---------------------------------------------------------------------------

class TicketSchema(BaseModel):
    """The validated ticket the model must produce.

    Field descriptions are prompt content read on every call.
    """
    intent: IntentType = Field(
        ...,
        description=(
            "The kind of request. Use 'order' for shipping/status questions, "
            "'refund' for return/refund requests, 'billing' for invoice/charge questions."
        ),
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        ...,
        description=(
            "Ticket urgency. Use 'critical' ONLY for outages that block users "
            "from receiving their order. Use 'high' for delivery delays. "
            "Use 'medium' for non-blocking inconveniences. Use 'low' for FYI."
        ),
    )
    customer_id: int | None = Field(
        None,
        description="Customer ID if known. May be None for first-time contacts.",
    )
    attachments: list[Attachment] = Field(
        default_factory=list,
        description="Files referenced in the intake message.",
    )


# ---------------------------------------------------------------------------
# Inbound HTTP body
# ---------------------------------------------------------------------------

class IntakeRequest(BaseModel):
    message: str = Field(..., min_length=1)
    customer_id: int | None = None
    provider: Literal["openai", "nano"] = Field(
        "openai",
        description=(
            "Which model to use. "
            "'openai' = gpt-5.4-mini-2026-03-17 (full reasoning). "
            "'nano'   = gpt-5.4-nano-2026-03-17 (small & fast)."
        ),
    )


# ---------------------------------------------------------------------------
# Routing tool result
# ---------------------------------------------------------------------------

class RoutingResult(BaseModel):
    team: str
    priority: str
    ticket_url: str
