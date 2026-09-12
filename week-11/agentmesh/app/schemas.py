"""Pydantic schemas - the contracts on the wire.

Two layers:
  - Skill I/O models (TriageInput, TriageOutput) - the per-skill schemas the
    Agent Card advertises. This is what the agent actually does.
  - A2A v1.0 protocol models (AgentCard, TaskState, TaskEvent, ...) - the wire
    format every A2A client on the other side already understands.

When a request fails validation here, FastAPI hands back a 422 with field-level
detail. That structured error is what the orchestrator on the other side reads to
decide whether to retry.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field


# ---------- Skill I/O (the agent's actual contract) ----------


class IncidentSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriageInput(BaseModel):
    """Input to the triage_incident skill."""

    description: str = Field(..., min_length=10, description="Free-text incident description")
    severity: IncidentSeverity
    user_id: str


class TriageOutput(BaseModel):
    """Output from the triage_incident skill.

    The agent's terminal payload goes through this model before it reaches the
    wire - an unvalidated dict is how a contract quietly rots.
    """

    category: Literal["knowledge", "action", "escalate"]
    summary: str
    proposed_action: str | None = None
    citations: list[str] = Field(default_factory=list)


# ---------- A2A v1.0 protocol models (the wire format) ----------


class TaskState(str, Enum):
    """The A2A v1.0 task states. THE VALUE IS THE STRING ON THE WIRE - there is no
    lowercase form in v1.0. Nine states; this flow drives seven of them.
    """

    TASK_STATE_UNSPECIFIED = "TASK_STATE_UNSPECIFIED"      # sentinel - NEVER entered
    TASK_STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
    TASK_STATE_WORKING = "TASK_STATE_WORKING"
    TASK_STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
    TASK_STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"  # declared; we authenticate at the door
    TASK_STATE_COMPLETED = "TASK_STATE_COMPLETED"
    TASK_STATE_FAILED = "TASK_STATE_FAILED"
    TASK_STATE_CANCELED = "TASK_STATE_CANCELED"
    TASK_STATE_REJECTED = "TASK_STATE_REJECTED"


#: A task in one of these is finished. The stream closes; no further events.
TERMINAL_STATES = frozenset(
    {
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)

#: Paused, waiting on something outside the agent. Not terminal.
INTERRUPTED_STATES = frozenset(
    {
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    }
)

#: The seven states this flow actually drives. UNSPECIFIED is a sentinel no code
#: path may assign; AUTH_REQUIRED has no producer here because authentication
#: happens at the door (JWKS), not mid-task.
DRIVEN_STATES = frozenset(
    {
        TaskState.TASK_STATE_SUBMITTED,
        TaskState.TASK_STATE_WORKING,
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)


class AgentSkill(BaseModel):
    """One thing this agent can be asked to do. `skills[]` is where an A2A v1.0 card
    describes its work - NOT `capabilities`, which is reserved for protocol flags.
    """

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    inputModes: list[str] = Field(default_factory=lambda: ["application/json"])   # media types
    outputModes: list[str] = Field(default_factory=lambda: ["application/json"])


class AgentCapabilities(BaseModel):
    """PROTOCOL FLAGS ONLY. Not a list of skills, not latency, not cost.

    A2A v1.0 defines exactly four: streaming, pushNotifications, extensions and
    extendedAgentCard. `stateTransitionHistory` was REMOVED in v1.0 (#1396); any
    tutorial still showing it predates March 2026.
    """

    streaming: bool = True
    pushNotifications: bool = False
    # Declared empty. A2A extensions are named protocol add-ons a caller can
    # detect before it commits; this build ships none.
    extensions: list[dict[str, Any]] = Field(default_factory=list)
    extendedAgentCard: bool = False


class AgentProvider(BaseModel):
    """Who operates this agent. In A2A this is how a caller learns whose service
    they are about to invoke - for the cohort, that is the student's name."""

    organization: str
    url: str


class AgentInterface(BaseModel):
    """One way to reach this agent. A2A v1.0 replaced the old single `url` plus
    `preferredTransport` pair with `supportedInterfaces[]`, an ORDERED list: the
    first entry is the preferred one, and a caller that cannot speak it reads
    further down rather than giving up.

    `protocolBinding` is an OPEN STRING, not an enum. The officially supported
    values are JSONRPC, GRPC and HTTP+JSON, and the field is deliberately open so
    other bindings can be added without a spec revision.
    """

    url: str
    protocolBinding: str = "HTTP+JSON"
    protocolVersion: str = "1.0"
    # Multi-tenant routing. A client that reads a tenant here MUST echo it back on
    # every request. Unused by this build.
    tenant: str | None = None


class AgentCardSignature(BaseModel):
    """One JWS signature over the card. A2A v1.0 carries `signatures[]`, a LIST,
    not a single `signature` - because key rotation means a card may legitimately
    carry two at once while the old key is still trusted.

    The signed content is the card with `signatures` itself removed, canonicalised
    per JCS (RFC 8785). This build declares the field and never populates it.
    """

    protected: str
    signature: str
    header: dict[str, Any] | None = None


class AgentEndpoints(BaseModel):
    """NOT AN A2A FIELD. This object is local to this build.

    A conformant v1.0 card advertises reachability through `supportedInterfaces[]`
    and nothing else; the route paths underneath are fixed by the binding, so a
    card has no business listing them. This build keeps `endpoints` because its
    routes are NOT the v1.0 REST binding (see the conformance table in the course
    notes) and its own browser UI and client script read it. Treat it as scaffolding
    you would delete on the way to conformance, not as something to copy.
    """

    base: str
    tasks: str = "/tasks"
    # The snapshot (A2A `tasks/get`). Advertised alongside the stream on purpose: a
    # caller that cannot hold a streaming connection - or whose network will not carry
    # one - needs to discover that polling is available, not guess at the path.
    status: str = "/tasks/{task_id}"
    stream: str = "/tasks/{task_id}/stream"
    input: str = "/tasks/{task_id}/input"
    # Set only when the MCP surface is co-hosted on this same origin, so a caller
    # is never pointed at a port that is really a separate process.
    mcp: str | None = None


class AgentCard(BaseModel):
    """The discoverable JSON document served at /.well-known/agent-card.json
    (legacy alias: /.well-known/agent.json)."""

    name: str = "agentmesh-triage"
    version: str = "1.0.0"
    protocolVersion: str = "1.0"
    description: str
    # NOT a v1.0 field. v0.x cards carried a single `url`; v1.0 replaced it with
    # `supportedInterfaces[]`. Retained here only so this build's own UI and
    # client script keep working, and kept in sync with supportedInterfaces[0].
    url: str
    provider: AgentProvider | None = None
    skills: list[AgentSkill]
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    securitySchemes: dict[str, Any] = Field(
        default_factory=lambda: {
            "bearer": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "scopes": ["triage:invoke"],
            }
        }
    )
    defaultInputModes: list[str] = Field(default_factory=lambda: ["application/json"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["application/json"])
    # REQUIRED in v1.0, and ordered: entry zero is this agent's preferred interface.
    # This is the field a conformant client reads to decide how to reach you.
    supportedInterfaces: list[AgentInterface]
    # Local scaffolding, not a spec field. See AgentEndpoints.
    endpoints: AgentEndpoints
    # Declared and deliberately unpopulated: this build does not sign its card and
    # does not verify anyone else's. Signing is a real A2A feature; pretending to
    # implement it would be worse than naming it as mocked. Note the field is
    # `signatures` PLURAL - a list - because key rotation needs two at once.
    signatures: list[AgentCardSignature] = Field(default_factory=list)


class TaskSubmit(BaseModel):
    """POST /tasks body. `skill` is the v1.0 field; `capability` is accepted as a
    legacy alias so a pre-v1.0 client is not broken by the rename."""

    skill: str = Field(validation_alias=AliasChoices("skill", "capability"))
    input: dict[str, Any]


class TaskAck(BaseModel):
    """Response to a successful POST /tasks."""

    task_id: str
    state: TaskState
    stream_url: str


class TaskEvent(BaseModel):
    """One SSE event on the wire."""

    event_id: int
    task_id: str
    state: TaskState
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(BaseModel):
    """GET /tasks/{task_id} - a point-in-time snapshot, A2A's `tasks/get`.

    The stream is the efficient way to watch a task; this is the reliable one. A
    caller that never opened a stream, lost one, or sits behind a proxy that will
    not forward a streaming body can still ask "where is my task?" and get the same
    state and payload the stream would have carried.
    """

    task_id: str
    state: TaskState
    timestamp: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
    last_event_id: int
    # True once no further events will ever arrive. An interrupted task is NOT final:
    # it is waiting for the caller, and will move again once they reply.
    final: bool
    interrupted: bool


class TaskInputReply(BaseModel):
    """POST /tasks/{task_id}/input - the caller's reply while the task is paused in
    TASK_STATE_INPUT_REQUIRED."""

    approved: bool
    note: str | None = None


class WhoAmIOutput(BaseModel):
    """Terminal payload of the `whoami` skill. Validated before it goes on the
    wire, same as TriageOutput - an identity answer is still a contract."""

    student_id: str
    student_name: str
    specialty: str
    agent_name: str
    base_url: str
    protocols: list[str] = Field(default_factory=list)
    # Echoed back so a caller can confirm which identity the server saw for THEM.
    caller: str | None = None


class TaskFailedPayload(BaseModel):
    """Payload attached to a TASK_STATE_FAILED event. A failure is a state with
    structure, not a bare 500: the orchestrator switches on `reason` and `retryable`."""

    reason: str
    evidence: str = ""
    retryable: bool = False
