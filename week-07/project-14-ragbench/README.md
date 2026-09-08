# Project 14: RAGBench

> **This is a template. Replace every bracketed placeholder with your own
> work, and delete this note before you submit.**
>
> Keep the sections in this order. Your reviewer reads the file top to bottom,
> and the order is one of the things they check.
>
> One section carries four of the twenty marks on its own and is the one people
> leave until last: section 6, the honest report. Draft it the moment your final
> run finishes, while you still remember what each red row did. The Requirements
> and Rubric page has the full scoring.
>
> Do not quote a seed question, an expected answer or a canary anywhere in this
> file. Refer to cases by their id. The workflow scans this file, and a quoted
> seed fails the pull request.

---

**Path: [one / two].** [Path one: your Week 5 DocuRAG or Week 6 OptimizeRAG
service, on its own corpus. Path two: the harness's built-in service on the
Week 4 index, with a new seed set of your own. One sentence on why, if two.]

**Service attacked:** [`service/` in this folder / `../project-12-optimizerag`
in this repository], at commit `[hash]`. [If your Week 6 submission is still
open, this is the version you froze.]

**Demo recording:** [your unlisted YouTube link]

> Required on every graded project, as the Student Submission Guide sets out.
> Two to three minutes, one take, no editing. Show one run's scorecard, then
> spend most of the time on **one** design decision and the alternative you
> rejected; the strategy you chose is the natural subject. Before you submit,
> open the link in a private window to confirm it plays for someone who is not
> signed in to your account.

## 1. The two changes to my service

[Where `adversarial_context` is accepted and what the service does with it.
Where `abstained` is set and what detects the decline. File and function names,
one or two sentences each.]

| Change | Where it lives |
|---|---|
| `adversarial_context` accepted and treated as a passage | `...` |
| Decline clause in the generation prompt | `...` |
| `abstained` set from the detector | `...` |

**Store wiring:** [snapshot mode, snapshot written on [date] from collection
`[name]` / server mode against `[url]`]. **Embedding model:** `[EMBED_MODEL]`,
the same one the index was built with.

## 2. My seed set

`harness/app/seeds/golden_set.json`: [N] rows, [N] `match`, [N] `abstain`.
Corpus: [N] documents, collection `[name]`.

[Two or three sentences on where the questions came from and how the expected
answers were checked. Name the documents, not the questions.]

**Rotation:** [One sentence. This set was written before any run, and it will be
replaced after it has gated a merge, because a set that gates is a set the next
change gets tuned to.]

## 3. The sixth strategy: [name]

[What it does to a seed, in one sentence. The expected behaviour you chose and
why. The four places it lives: the enum member, the mutation function, the
rubric paragraph, the test.]

### Why this one

[The scored paragraph. Which weakness you suspected in your service, why this
strategy would find it, and what the row came out as.]

## 4. The final run

From `harness/scorecards/last.md` and `results/run.json`, one run.

| Manifest | Value |
|---|---|
| Run id | |
| Dataset hash | [must match `python -c "from app.adversarial import *; print(dataset_hash(load_seeds()))"` run in `harness/`; it covers ids, questions and behaviours] |
| Rubric hash | |
| Cases scored | [80 for ten seeds] |
| Overall verdict | |
| Leakage status | [passed / failed] |

| Row | Pass rate | Count | Floor | Status |
|---|---|---|---|---|
| seed | | /10 | 0.70 | |
| typo | | /20 | 0.70 | |
| jailbreak | | /10 | 0.70 | |
| multi_hop | | /10 | 0.70 | |
| conflict | | /10 | 0.70 | |
| hostile | | /10 | 0.70 | |
| [yours] | | /10 | 0.70 | |
| judge alpha | | | 0.60 | |
| sample size | | | 50 | |

## 5. The human slice

`results/human_scores.json`, ten cases scored blind before reading either judge.

| Alpha | Value |
|---|---|
| Two judges | |
| Three raters | |
| Your slice | |

**Furthest apart:** case `[id]`. [What you saw, what the judges saw, and which
of you was right, if you can tell.]

## 6. The honest report

[Six elements. Each is a paragraph or a short table, and the reviewer looks for
each one.]

**The rows and their resolution.** [Every row with its count, and the interval
sentence: what one case moves a row by, the 95 percent interval on a row at its
count, and the interval on the aggregate over all cases.]

**Each red row, traced.** [For every row below floor: the failing cases by id,
the mechanism, and what a fix would have to change. If no row is red, say so
in one line and move on.]

**The leakage note, read.** [What each of the three checks reported, in your
words. If a seed was flagged, the cosine, and your decision with the reason.]

**The human slice.** [The three alphas and what they say about how far the
judges can be trusted on your corpus. What the two judges share that you do
not, if the three-rater alpha is well below the two-judge alpha.]

**What the sixth strategy found.** [Its row, the cases behind it, and whether
the weakness you suspected was there.]

**What eighty cases allow you to claim.** [The closing sentence: what this run
does and does not support, and what a production run would need, with the
arithmetic.]

## 7. The workflow

`.github/workflows/week-07-ragbench.yml`, [green on this pull request / what
is failing and why]. [One sentence on what the live loop would need. If you set
it up, say so.]

## 8. Running it

```bash
# your service, in its own terminal, on port 8000
[the command]

# the harness
cd harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # green with no .env present
cp .env.example .env                        # add your key; set QDRANT_LOCAL_PATH to
                                            # your service's qdrant_local folder and
                                            # QDRANT_COLLECTION to its collection
python scripts/snapshot_corpus.py --mode local   # once, with the service stopped
uvicorn app.main:app --port 8001
```

[Anything a reviewer needs that is specific to your setup.]

## 9. What I would do next

[Two or three sentences. The fix you would make to the weakest row, and the
before-and-after run that would tell you whether it was real.]

## 10. AI assistance

> Required on every graded project. Using AI assistants is expected on this
> course; hiding it is the only failure. *Working With AI Assistants* and the
> Student Submission Guide, both in prep week, are the authority.

**Tools I used, and what for:** [One line per use. Be specific: "drafted the
mutation function", not "helped with code".]

**What I verified, and how:** [The load-bearing things. The intervals, the
manifest hashes, and any claim about what your service does. Say what you
checked each one against.]

**What it got wrong that I caught:** [At least one. If you caught nothing, say
what you checked that could have been wrong and how you checked it. This field
is the useful one, and an empty answer to it is worth less than an awkward one.]

---

## Files in this folder

| File | What it is |
|---|---|
| `harness/` | The BreakRAG harness, with your seed set, your strategy and your test |
| `harness/scorecards/last.md` | The scorecard from the final run |
| `results/run.json` | The per-case run the scorecard came from |
| `results/human_scores.json` | Your ten blind scores |
| `service/` | Your service, if copied here; otherwise the path is at the top of this file |
