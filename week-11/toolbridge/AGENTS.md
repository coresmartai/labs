# AGENTS.md

Forty lines, written by hand, containing only what this repository cannot tell
you itself. See W11-R03 for why the architecture overview is deliberately absent.

## Commands

- Tests: `pytest -q`. Twenty-seven of them, no network, under two seconds.
- The peer check: `python scripts/peer_up.py`. Run this before anything else.
- The eval: `python -m app.eval`. Needs `OPENAI_API_KEY`. Nothing else does.
- Drive the server for real: `python scripts/mcp_client_demo.py stdio`.

## The thing that will waste your afternoon

This project pins `mcp==2.2.0`. The two guided builds in Section 3 pin
`mcp==1.28.1`. **They are different majors and they need separate virtual
environments.** If you see `ModuleNotFoundError: No module named
'mcp.server.fastmcp'` you are in the right venv and reading 1.x code. If you see
`AttributeError: 'Tool' object has no attribute 'inputSchema'` you are reading a
wire field name off a Python object; it is `input_schema` in 2.x.

## Conventions that differ from the obvious

- Tool bodies in `app/tools.py` import nothing from `mcp`. `app/mcp_server.py`
  does the registering. Keep it that way: it is why the tests need no server.
- Every failure goes through `app/errors.fail()`. One shape, one place.
  `retryable` is a convention we define, not something the protocol mandates.
- `app/a2a.py` ships complete. If you are editing it to make a test pass, the
  test is probably telling you something about your tool instead.
- Descriptions live in `app/descriptions.py` and are reviewed like prose,
  because that is what they are.

## Never

- Never commit `.env`. `PEER_TOKEN` is a credential even when it is `demo-token`.
- Never resubmit a task after a dropped stream. Re-attach from the cursor.
- Never treat `TASK_STATE_REJECTED` as retryable.
