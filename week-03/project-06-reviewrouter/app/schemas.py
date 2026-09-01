"""The shapes.

The ENUMS are given to you. The DESCRIPTIONS are not, and they are worth
more marks than any code in this repository.

A description string is prompt content. The model re-reads it on every
single call. "The kind of problem" is true and useless. A description that
decides the boundary case is what separates 90% accuracy from 60%.

Two pairs overlap on purpose:
    defect vs usability   - broken, or merely bad?
    billing vs delivery   - charged for something that never arrived
Your descriptions have to settle both, and your tie-break rule has to
settle a review that raises two problems at once.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    billing = "billing"
    defect = "defect"
    delivery = "delivery"
    usability = "usability"
    praise = "praise"
    other = "other"


class Priority(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"
    critical = "critical"


class OrderLookupArgs(BaseModel):
    """Arguments the model sends when it wants an order looked up."""

    order_id: str = Field(
        ...,
        description="TODO: what counts as an order reference, and what the "
        "model should do when the review mentions a number that is clearly "
        "not one.",
    )


class Ticket(BaseModel):
    """The object your service produces for one review.

    Field ORDER matters. Generation runs top to bottom and providers emit
    properties in the order the schema declares them, so a reasoning field
    placed after the verdict is a rationalisation you paid for. If you add
    one, put it first.
    """

    type: Intent = Field(
        ...,
        description="TODO. This is the single highest-leverage string in the "
        "project. Say where defect ends and usability begins, and where "
        "billing ends and delivery begins. State your tie-break rule here so "
        "the model can actually apply it.",
    )
    order_id: Optional[str] = Field(
        None,
        description="TODO. Optional on purpose: many reviews mention no order "
        "at all. Give the model a legal way to say it did not find one, so it "
        "does not invent a plausible-looking number.",
    )
    priority: Priority = Field(
        ...,
        description="TODO. Define these by CONSEQUENCE, not by the customer's "
        "tone. If your description lets exclamation marks raise the priority, "
        "almost everything will come back critical.",
    )


class RoutedTicket(BaseModel):
    """What your code produces after the schema gate and the routing table."""

    ticket: Ticket
    team: Optional[str] = Field(
        None,
        description="Filled in by your code from routing.route_to_team. "
        "None is a valid, correct answer for praise.",
    )
