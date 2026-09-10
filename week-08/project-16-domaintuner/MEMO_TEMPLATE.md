# MEMO.md template

Copy everything below the line into `MEMO.md` and fill it in. This file is rubric lines 5 and 8,
five of the twenty marks, and it is the deliverable of the project.

---

# DomainTuner decision memo: <your domain>

## The table

| approach | accuracy | p50 ms | p95 ms | cost/1k | trainable | privacy |
|---|---|---|---|---|---|---|
| base | | | | | 0 | in_perimeter |
| tuned | | | | | | in_perimeter |

Every number here comes from `results/base.json` or `results/tuned.json`. If a figure is not in one
of those files, take it out of this table.

Per-category breakdown, if your evaluation set has categories:

| category | base | tuned | moved |
|---|---|---|---|

## What I would ship, and for what use

Not which one scored highest. Which one you would put in front of users, given accuracy, latency,
cost and where the data sits.

## What would change that

Name a specific condition: a dataset of a particular size, a latency budget, a rule about data
leaving your systems, a different scoring method. Not a general caveat.

## What these numbers do not support

Your sample size, your scoring rule, and how many runs sit behind each row. Say what that means and
what it does not.

---

**The commonest way to lose marks here** is to write three sections that would be true of anybody's
project. Name your own rows, your own categories and your own figures.
