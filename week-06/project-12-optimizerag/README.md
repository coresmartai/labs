# Project 12: OptimizeRAG

> **This is a template. Replace every bracketed placeholder with your own
> work, and delete this note before you submit.**
>
> Keep the sections in this order. Your reviewer reads the file top to bottom,
> and the order is one of the things they check.
>
> Two sections carry most of the marks and are the two people leave until last:
> section 5, where you name the rows that changed, and section 6, the honest
> read. Draft both as soon as your comparison run finishes, while you still
> remember why each row moved. The Requirements and Rubric page has the full
> scoring.

---

**Corpus path: [one / two].** [One sentence. Path one means your Week 5 DocuRAG
corpus and golden set. Path two means a corpus you ingested this week, and you
should say why.]

**Week 5 service:** [copied into this folder / imported from `../project-10-docurag`].
[If your Week 5 submission is still open, give the commit hash you froze as the
baseline.]

**Demo recording:** [your unlisted YouTube link]

> Required on every graded project, as the Student Submission Guide sets out.
> Two to three minutes, one take, no editing. Show it working on real input,
> then spend most of the time on **one** design decision and the alternative you
> rejected. Before you submit, open the link in a private window to confirm it
> plays for someone who is not signed in to your account.

## 1. What I changed

[Which two or three techniques you added, in one or two sentences each. Name the
file in this folder where each one lives.]

| Technique | Added? | Where it lives |
|---|---|---|
| HyDE query transformation | yes / no | `...` |
| Cross-encoder reranking | yes / no | `...` |
| Extractive compression | yes / no | `...` |

### Why these ones

[The scored paragraph. Is your problem a query problem or an index problem?
What in your corpus made you think so? If the honest answer is that your chunks
lost their context at ingestion, say that, name Contextual Retrieval as the fix
that would address it, and explain why you are measuring a query-side change
this week anyway.]

## 2. The setup, so the comparison is countable

| What | Value |
|---|---|
| Golden set | `data/golden_dataset.json`, [N] rows, [N] answerable, [N] refusal |
| Corpus | [N] documents, collection `[name]` |
| Similarity threshold | [value] |
| Spread threshold | [value] |
| Wide candidates / injected chunks | [30] / [5] |
| Baseline gates on | the raw question |
| After-pipeline gates on | [the raw question and the probe / the raw question] |

[If you are on path two, add one sentence saying how you placed these
thresholds and confirming they did not change between the two runs.]

## 3. Before and after: the five product metrics

From `results_compare.json`, one run.

| Metric | Baseline | Optimised | Delta |
|---|---|---|---|
| Groundedness | | | |
| Citation validity | [rate] ([N] generated) | [rate] ([N] generated) | |
| Citation recall | | | |
| False-answer rate | | | |
| False-refusal rate | | | |

## 4. Before and after: the four retrieval numbers

| Number | Baseline | Optimised | Delta |
|---|---|---|---|
| Recall at five | | | |
| p50 latency (ms) | | | |
| p95 latency (ms) | | | |
| Tokens per query | | | |

**Recall denominator:** [N] answerable rows. Refusal rows are excluded because
they have no chunk to recall.

**How I established the recall ground truth:** [One or two sentences. Which
chunk counts as correct for each answerable row, and how you determined it.]

**Warm-up:** [one warm-up call before timing / first row dropped].

## 5. The rows that moved

| Row | Question (short) | Baseline | Optimised | Why it changed |
|---|---|---|---|---|
| | | | | |

[Every row where the two columns differ. If no rows differ, say so here in one
line: that is a real result and it is worth full marks when it is explained.]

## 6. The honest read

[One paragraph. It states all nine numbers with direction and size. It puts the
sample size in the same sentence as the improvement. It names the rows that
moved, by number, and says why they moved. It gives the denominators wherever
the two columns differ. It closes with a sentence on what this sample does and
does not let you conclude.]

## 7. Architecture

![My OptimizeRAG pipeline](docs/architecture.png)

[A few sentences describing the diagram, so that someone using a screen reader
gets the same information: the stages in order, the counts at each stage, and
where the gate sits.]

## 8. Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                 # green with no .env present
cp .env.example .env      # then fill in your values
uvicorn app.main:app --reload --port 8000
```

[Anything a reviewer needs that is specific to your setup.]

## 9. What I would do next

[Two or three sentences. The change you would make with another week, and the
measurement that would tell you whether it worked. A technique you tried and
rejected belongs here, with the number that made you reject it.]

## 10. AI assistance

> Required on every graded project. Using AI assistants is expected on this
> course; hiding it is the only failure. *Working With AI Assistants* and the
> Student Submission Guide, both in prep week, are the authority.

**Tools I used, and what for:** [One line per use. Be specific: "drafted the
union-merge function", not "helped with code".]

**What I verified, and how:** [The load-bearing things. Numbers, API calls, and
any claim about what your code does. Say what you checked each one against.]

**What it got wrong that I caught:** [At least one. If you genuinely caught
nothing, say what you checked that could have been wrong and how you checked it.
This field is the useful one, and an empty answer to it is worth less than an
awkward one.]

---

## Files in this folder

| File | What it is |
|---|---|
| `results_baseline.json` | The Week 5 pipeline, gate active, over the golden set |
| `results_compare.json` | Both pipelines, one run, five metrics per column plus per-row detail |
| `data/golden_dataset.json` | Your golden set, with recall ground truth |
| `docs/architecture.png` | The diagram from section 7 |
