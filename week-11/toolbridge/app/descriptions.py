"""TASK 1. The tool descriptions.

The description is the prompt the model reads to decide whether to call a tool.
It is the highest-leverage thing in this repository and it is prose, which is
why it lives in its own file and gets reviewed like one.

Two sets. `good` is what you write. `bad` is deliberately vague, and exists so
that `python -m app.eval` gives you two numbers rather than one opinion.

The five moves, from W11-R01:
  1. Verb first          Search, Fetch, Submit, Resume
  2. Scope explicit      which peer, whose incidents, what range
  3. Trigger stated      "Use when the user asks ..."
  4. Output declared     what comes back, and in what shape
  5. Edge cases listed   what an empty result looks like, what is NOT an error
"""
from __future__ import annotations

# TASK 1: write these three descriptions.
#
# This is the highest-leverage work in the repository and it is prose. Spend
# real time here. The tests check that you made the five moves; the eval checks
# whether a model can actually route on what you wrote.
#
# A worked example, from W11-R01, in three passes:
#   pass 1  "Cancels a deployment."                     <- true and useless
#   pass 2  + when to fire, and whose deployments       <- better
#   pass 3  + what comes back, and what the edge is     <- the version that works
#
# Aim for 200 characters or more per description. If you cannot describe the
# tool cleanly in a paragraph, it is doing too much.
GOOD = {
    "peer_card": "",
    "delegate_triage": "",
    "resume_task": "",
}


# What the same three tools look like when somebody writes the description after
# writing the implementation. Every one of these is TRUE and every one of them
# is useless: no verb, no trigger, no output shape, no edges.
BAD = {
    "peer_card": "Gets agent data.",
    "delegate_triage": "Handles incidents with the other agent.",
    "resume_task": "Continues a task.",
}


def describe(name: str, quality: str) -> str:
    return (GOOD if quality == "good" else BAD)[name]
