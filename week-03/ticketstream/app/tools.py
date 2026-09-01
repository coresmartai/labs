"""TicketStream tools.

`extract_ticket` is the model-facing tool that produces the structured ticket.
`route_to_team` is the side-effect tool - NEVER called directly by the model;
only invoked by our code AFTER the Pydantic gate passes.
"""
import time

from app.schemas import TicketSchema, RoutingResult


# ---------------------------------------------------------------------------
# Tool: extract_ticket (model-facing)
# ---------------------------------------------------------------------------

EXTRACT_TICKET_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_ticket",
        "description": (
            "Extract a structured Ticket from the user's intake message. "
            "Always call this tool. Do not respond in prose."
        ),
        "parameters": TicketSchema.model_json_schema(),
    },
}


# ---------------------------------------------------------------------------
# Tool: route_to_team (NOT model-facing - invoked by our code)
# ---------------------------------------------------------------------------

def route_to_team(team: str, priority: str) -> RoutingResult:
    """Side-effect tool: route the validated ticket to a team queue.

    NOT in the model's tool list. Our code calls it after TicketSchema validates.
    This is the safety pattern: side effects gated behind validated intent.
    """
    time.sleep(0.05)  # simulated network round-trip
    return RoutingResult(
        team=team,
        priority=priority,
        ticket_url=f"https://tickets.internal/{team}/{int(time.time())}",
    )


def team_for_intent(ticket: TicketSchema) -> str:
    """Map intent type → team queue. Deterministic, testable, in code."""
    intent_type = ticket.intent.type
    return {
        "order": "fulfillment",
        "refund": "refunds",
        "billing": "billing",
    }[intent_type]


