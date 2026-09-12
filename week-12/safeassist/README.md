# SafeAssist - Week 12, Project 24

A guarded clinical assistant with a human in the loop. **20 marks.**

You start from GuardianAI rather than an empty directory. The retrieval stack,
the access rings, the audit log and three of the four Presidio call sites are
given and working. Six things are not, and they are yours.

**The suite is the specification.** A task is done when its tests go green.

```
pytest -q        # 19 failed, 9 passed  <- the intended starting state
```

That is not a broken download. The failures are the assignment.

---

## Setup

Identical to GuardianAI, with one addition.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg     # ~450 MB. Not optional.
cp .env.example .env                        # set OPENAI_API_KEY
docker run -d -p 6333:6333 qdrant/qdrant    # or a free Qdrant Cloud cluster
python -m app.ingest --reset
uvicorn app.main:app --reload --reload-dir app
```

If `pip` refuses to resolve, you are in a virtual environment left over from an
earlier week. This package needs **pydantic 2.12.5 or newer**, because Presidio
2.2.364 requires it, and the resolver error does not name Presidio as the cause.

## The domain

A fictional clinic. Twenty seed documents, three roles (`analyst`, `clinician`,
`audit`), and one domain identifier: a medical record number, `MRN-4417-2026`.

**Nothing in this repository is real, and nothing real may be added to it.**
The cohort repository is shared and the Week 7 project scans all of it.
Fixtures are generated at test time by `fixtures/generate.py`, are seeded so a
failure is reproducible, carry a canary token, and are gitignored. A submission
whose fixtures are committed, or whose fixtures lack the canary, fails that
criterion however good the rest of the code is.

---

## The six tasks

### TASK 1. Fix the ordering defect. **4 marks**

`app/main.py` scrubs the query and then classifies the scrubbed copy. Reading 3
has the trace. The short version: the scrubber replaces a token the classifier's
pattern depends on, so the classifier returns not flagged, no refusal fires, and
the instruction reaches the model.

Reproduce it before you fix it:

```bash
pytest -q -k task1
```

Fix it so the classifier sees the **raw** query and the scrubber runs on its own
copy for egress. Keep both audit events: an attack that also carried personal
data must produce an `injection_refusal` **and** a `pii_scrub`.

Do not delete the scrub. Minimisation still has to happen.

### TASK 2. Register the MRN recogniser. **3 marks**

`app/presidio_layer.py`. Word boundaries, not anchors, because this is a scanner
and not a validator. A base score that clears the redaction threshold on its
own. A context list. And add `MRN` to the startup probe's required set, because
a probe that does not check a control is a control the boot will start without.

### TASK 3. Close the fifth egress path. **3 marks**

`app/telemetry.py` is written and is never called. Emit one telemetry line per
answered request from `app/main.py`.

The tests check the **fields** as well as the message. Structured logging puts
the interesting values in keyword arguments, and a scrubber applied only to the
message string is applied to the least likely place for a value to be.

### TASK 4. Build the human-review queue. **4 marks**

`app/review.py`. Three functions raise `NotImplementedError`. The record shape,
the masking helper, `load()` and `stats()` are given; the transitions are yours.

- `enqueue` creates a pending record. It must never carry the detected value.
- `decide` moves a pending record to a terminal state. A decided record cannot
  be decided again: raise `ValueError` rather than overwrite. An audit trail in
  which a decision can be replaced is not an audit trail.
- `sweep` escalates records older than `review_ttl_hours`. A queue with no
  timeout is a backlog nobody has noticed yet.

Then wire it: in `app/main.py`, a detection in the log-only band must create a
record. This is the task that makes the week's title true, and Week 13 renders
approval buttons over these four routes.

### TASK 5. Measure the detector on your own data. **4 marks**

Build `data/pii_labelled.json`. `data/pii_labelled.example.json` is the shape;
you need at least ten cases, and they must include **negatives** and **hard**
cases, because a set made only of easy positives cannot move either metric.

Report precision and recall per entity with `app/pii_eval.py`. Then apply the
Week 5 and 6 rule and write the answer in your memo: **for each metric, what
input would make it drop?** If the answer is "nothing", the metric is decoration
and you have rebuilt the `citation_precision` bug with a new label.

Expect an uncomfortable number somewhere. That is the point.

### TASK 6. Complete the responsible-AI checklist. **1 mark**

Rename `RESPONSIBLE_AI.template.md` to `RESPONSIBLE_AI.md` and fill it in. A
ticked item must name a file in backticks and that file must exist.

**Leaving items unticked is expected and costs nothing.** For each one you leave
unticked, write a sentence saying why. A checklist where everything is ticked
after one afternoon was filled in rather than run.

### The memo. **1 mark**

Four hundred words. Two questions:

1. Of the controls in this service, which is a **guarantee** and which is a
   **mitigation**? Name one of each and say how you can tell.
2. What would you do with one more week, and why that rather than something else?

---

## What good looks like

```
pytest -q        # 28 passed
```

Plus a pull request with the README template filled in, the eval report, the
checklist, and the memo.

## Deliberately not in scope

- Separating write access to the corpus from read access, which Reading 4
  identifies as a real gap. Worth a sentence in the memo; not worth marks here.
- Authenticating `POST /eval`. Same.
- A reviewer interface. That is Week 13's job, and your backend is what it needs.
