"""The canned corpus. Eight global IT policies, plus prior tickets per user.

Global policy chunks are scope="global", collection="policies".
Prior tickets are scope="user:{id}", collection="tickets".

The two are deliberately close in wording. A scoped search that ranks before it
filters will pull policy chunks into a per-user query and drop the tickets, which
is the whole point of the exercise in app/vectorstore.py.
"""
from __future__ import annotations

from typing import Any

POLICIES: list[dict[str, Any]] = [
    {"chunk_id": "pol-1", "source": "policies/access-requests.md",
     "text": "Access requests to production systems require the requester's manager to approve "
             "before the grant is applied. Standing access is reviewed every ninety days."},
    {"chunk_id": "pol-2", "source": "policies/access-requests.md",
     "text": "Read-only access to the analytics warehouse is self-service for full-time staff. "
             "Write access to any production database is never self-service."},
    {"chunk_id": "pol-3", "source": "policies/software.md",
     "text": "Software on the approved catalogue installs without review. Anything outside the "
             "catalogue needs a security assessment before it may be installed on a work device."},
    {"chunk_id": "pol-4", "source": "policies/software.md",
     "text": "Licence seats are charged to the requesting team's cost centre. A seat that has gone "
             "unused for sixty days is reclaimed automatically."},
    {"chunk_id": "pol-5", "source": "policies/hardware.md",
     "text": "Laptop replacement is on a thirty-six month cycle. Replacement before that needs a "
             "fault report or a documented change in role requirements."},
    {"chunk_id": "pol-6", "source": "policies/hardware.md",
     "text": "Monitors, keyboards and headsets are stock items and ship from the local depot "
             "without approval, up to one of each per person per year."},
    {"chunk_id": "pol-7", "source": "policies/offboarding.md",
     "text": "On the last working day all access is revoked, devices are returned, and the "
             "account is disabled rather than deleted so that audit records survive."},
    {"chunk_id": "pol-8", "source": "policies/security.md",
     "text": "Shared credentials are prohibited. Where a service cannot issue per-person "
             "credentials, access goes through the vault and every use is logged."},
]

# Prior tickets, per user. These are EPISODIC memory: this user's own history.
TICKETS: dict[str, list[dict[str, Any]]] = {
    "u_1": [
        {"chunk_id": "tkt-u1-1", "source": "tickets/DESK-4410",
         "text": "Requested write access to the billing database for a migration. Manager approved, "
                 "access granted for fourteen days and expired on schedule."},
        {"chunk_id": "tkt-u1-2", "source": "tickets/DESK-4712",
         "text": "Asked to install a non-catalogue profiler. Security assessment came back clear and "
                 "the install was approved with a note to re-review at the next major version."},
        {"chunk_id": "tkt-u1-3", "source": "tickets/DESK-5003",
         "text": "Laptop replaced early after a documented role change from support to engineering. "
                 "Old device returned the same week."},
    ],
    "u_2": [
        {"chunk_id": "tkt-u2-1", "source": "tickets/DESK-4588",
         "text": "Requested read-only warehouse access, which is self-service, and was pointed at the "
                 "portal rather than raising a ticket."},
        {"chunk_id": "tkt-u2-2", "source": "tickets/DESK-4901",
         "text": "Second monitor shipped from the depot. Within the annual stock-item allowance, so "
                 "no approval was needed."},
    ],
}
