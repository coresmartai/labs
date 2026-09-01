# ReviewRouter

Project 6, Week 3. Rename this file to `README.md` before you open the pull
request. The two headings below are graded, two marks between them, and they
are marked on the reasoning rather than on which choice you made.

## What this is

One paragraph. What the service does and how to run it.

## Running it

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your key in .env
uvicorn app.main:app --reload
pytest                        # must be green with no key set
```

## Model

Which model, pinned to a dated version, and why. One or two sentences.

## The tie-break rule  (GRADED)

Some reviews raise two problems at once. State which one wins, and why.

Write the rule as a rule, not as a description of what your code happens to
do, and then say what it costs you. Every tie-break rule is wrong for some
review; naming the case where yours is wrong is worth more than pretending
there isn't one.

A rule stated with no reason scores half marks.

## The refusal rule  (GRADED)

Sometimes `lookup_order` says the order does not exist, and sometimes the
review mentions no order at all. State what your service does in each case,
and why.

Things to actually decide, not to list: does the ticket still route? Does the
priority change? Does `order_id` come back null, or does the whole request
fail? What would you want to happen if this were your support queue at 2am?

## What I would do next

Optional, ungraded. One or two lines. What you would fix first with another
day on it.
