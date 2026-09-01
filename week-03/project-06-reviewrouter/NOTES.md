# Where to start

Read `W03-P06` first. It is the spec, and this repository is only the
scaffolding for it.

Order of work, roughly:

1. `app/schemas.py`. Write the field descriptions. Do this FIRST, before any
   wiring. Four marks of classification accuracy are downstream of these
   three strings, and they are much harder to write once you are deep in
   streaming code.
2. `app/tools.py`. Define `TOOL_DEFINITION`. The description must say when
   NOT to call the tool.
3. `app/llm.py`. The three functions, in order: `resolve_lookup`,
   `stream_ticket`, `validate_with_correction`.
4. `app/main.py`. Wire the stream. The tests in
   `tests/test_intake_contract.py` go green here.
5. `README.md`. The two rules.

## What is already done for you

- `app/routing.py` is complete. Do not change it.
- `app/config.py` is complete, including both caps.
- `reviews.jsonl` is your pool. Forty reviews, unlabelled.
- `tests/test_given.py` passes now and must keep passing.

## What is deliberately missing

There is no answer key in this repository. There is no worked solution. The
ambiguous reviews in the pool are ambiguous on purpose, and the two reviews
containing an injection attempt are there to find out whether your routing
tool is reachable from the model. It should not be.
