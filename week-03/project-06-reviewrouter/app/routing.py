"""The routing table.

This is GIVEN to you, filled in. It is not a prompt, it is not a sentence,
and the model never sees it. Your code calls route_to_team() after the
schema gate passes.

Note the asymmetry before you try to be clever: usability, praise and other
do NOT escalate. Anyone who writes "append -escalation when priority is high"
gets three of these six rows wrong.
"""
from typing import Optional

from app.schemas import Intent, Priority

_ESCALATED = {Priority.high, Priority.critical}

_TABLE: dict[Intent, tuple[Optional[str], Optional[str]]] = {
    #                  (low / normal,   high / critical)
    Intent.billing:   ("billing",      "billing-escalation"),
    Intent.defect:    ("engineering",  "engineering-oncall"),
    Intent.delivery:  ("fulfillment",  "fulfillment-escalation"),
    Intent.usability: ("product",      "product"),
    Intent.praise:    (None,           None),
    Intent.other:     ("triage",       "triage"),
}


def route_to_team(intent: Intent, priority: Priority) -> Optional[str]:
    """Return the owning team, or None when no action is required.

    NOT a tool. This function is never placed in the tool list sent to the
    model. It is called by your own code, after validation.
    """
    low, high = _TABLE[intent]
    return high if priority in _ESCALATED else low
