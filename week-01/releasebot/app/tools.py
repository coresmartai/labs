"""
Tool definitions + dispatch.

A "tool" is a function the model can choose to call. Each tool has:
  - a name
  - a JSON-schema input definition (so the model knows how to call it)
  - a Python implementation (so we know how to execute it)

Key design choice: the send_email tool accepts structured fields
(headline, bullets, risk_level) rather than a plain text body.
This means the tool call arguments ARE the structured output - no
second LLM call and no free-text JSON parsing needed.

Two questions to ask of every tool you ever ship:
  1. What happens if the model calls this twice? (idempotency)
  2. What's the blast radius if the model calls it incorrectly? (safety)
"""

from __future__ import annotations
from typing import Any
import logging
import smtplib
import ssl
from email.mime.text import MIMEText

from app.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Tool schema sent to the model (OpenAI function-calling format)
# ---------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Send a structured release summary by email to the team. "
                "Fill in all fields based on the release notes provided."
            ),
            # strict mode: the API constrains argument generation to this
            # schema - guaranteed parseable JSON, guaranteed shape and enum.
            # Strict mode does NOT support length/count keywords (maxLength,
            # minItems, ...), so Pydantic (ReleaseSummary) enforces those
            # at the boundary.
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address.",
                    },
                    "headline": {
                        "type": "string",
                        "description": "One-line release headline, max 120 chars.",
                    },
                    "bullets": {
                        "type": "array",
                        "description": "2-6 short bullet strings covering the key changes.",
                        "items": {"type": "string"},
                    },
                    "risk_level": {
                        "type": "string",
                        "description": "Operational risk: low = safe to ship, med = monitor, high = careful rollout.",
                        "enum": ["low", "med", "high"],
                    },
                },
                "required": ["to", "headline", "bullets", "risk_level"],
                "additionalProperties": False,
            },
        },
    }
]


# ---------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------

def send_email(to: str, headline: str, bullets: list[str], risk_level: str) -> dict[str, Any]:
    """
    Send a formatted release summary via Gmail SMTP (port 465, SSL).

    The email body is built from the structured fields - not from free text -
    so the format is always consistent.

    Setup: create a Gmail App Password at
    https://myaccount.google.com/apppasswords
    (requires 2FA enabled on the account).
    """
    settings = get_settings()

    subject  = f"Release Update: {headline[:80]}"
    body     = (
        f"{headline}\n\n"
        + "\n".join(f"• {b}" for b in bullets)
        + f"\n\nRisk level: {risk_level}"
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = settings.smtp_sender
    msg["To"]      = to

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(settings.smtp_sender, settings.smtp_password)
            server.sendmail(settings.smtp_sender, to, msg.as_string())
        logger.info("Email sent  to=%s  subject=%r", to, subject)
        return {"success": True, "to": to, "subject": subject}
    except Exception as e:  # SMTPException, socket.gaierror, ssl.SSLError, ConnectionRefusedError...
        logger.exception("Email send failed")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------

def execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Look up a tool by name and run it with the model-supplied args."""
    if name == "send_email":
        return send_email(
            to=args.get("to", ""),
            headline=args.get("headline", ""),
            bullets=args.get("bullets", []),
            risk_level=args.get("risk_level", "low"),
        )
    return {"success": False, "error": "unknown_tool", "tool": name}
