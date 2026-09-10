# ResearchAgent

**Applied GenAI & Agentic AI Engineering Course - Week 9, Project 18**

A bounded research loop over a fixed corpus of IETF email-authentication RFCs, producing a
cited brief. It is OpsAssist's loop with three things changed: the tools, the stop conditions,
and what a run produces.

**This is a scaffold, not a finished app.** Four things are yours to build. Everything else
runs as shipped.

---

## 0. What you are building

```
researchagent/
├── corpus/
│   ├── fetch.py             <- downloads 7 RFCs, hashes them, writes manifest.json
│   ├── manifest.json        <- YOU COMMIT THIS. The rfc/ folder you do not
│   └── rfc/                 <- gitignored. Never commit the corpus
├── app/
│   ├── config.py            <- typed settings. One TODO: your stop condition's settings
│   ├── schemas.py           <- the contract surface. Read this first
│   ├── chunker.py           <- RFC text to chunks, at runtime, page furniture stripped
│   ├── retrieval.py         <- BM25. Deterministic, local, no key
│   ├── ledger.py            <- THE PROJECT. Ships with a known-wrong implementation
│   ├── tools.py             <- search_corpus, read_chunk, list_documents + dispatcher
│   ├── llm.py               <- the provider seam, unchanged from OpsAssist
│   ├── agent.py             <- the research loop. One TODO: your stop condition
│   ├── brief.py             <- assembles a brief section from the ledger
│   ├── validator.py         <- deterministic citation checking. No model call
│   └── main.py              <- run three questions, write five artefacts
├── tests/
│   ├── test_ledger.py       <- four tests. Three FAIL until you build the ledger
│   ├── test_validator.py    <- what the validator catches
│   └── test_smoke.py        <- dispatcher contract, catalogue shape, licence check
├── questions.example.json   <- copy to questions.json and write your three
├── requirements.txt
└── .env.example
```

### The four things that are yours

| # | What | Where |
|---|---|---|
| 1 | **A ledger whose ids are stable across rounds** | `app/ledger.py`, then one line in `app/main.py` |
| 2 | **A stop condition that answers "is this research finished?"** | `app/agent.py` and `app/config.py` |
| 3 | **Three questions**, one answerable, one partial, one not answerable at all | `questions.json` |
| 4 | **The design memo** | your README |

---

## 1. Setup

Python **3.10 or newer**. On macOS the system `python3` is 3.9, which is not a broken install;
build the virtual environment with an explicit interpreter.

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put a real key in it
python corpus/fetch.py             # ~1 MB, seven files
pytest -q                          # expect 3 failed, 12 passed. That is correct
```

**Three failures on a clean install is the intended starting state.** `test_ledger.py` holds
four tests: one per property your ledger needs, plus a fourth that already passes. Read them
before you read anything else:

```bash
pytest -q tests/test_ledger.py
```

---

## 2. The corpus, and why it is fetched rather than committed

Seven RFCs on one topic:

| RFC | Title | Date |
|---|---|---|
| 7208 | Sender Policy Framework (SPF) v1 | Apr 2014 |
| 6376 | DomainKeys Identified Mail (DKIM) Signatures | Sep 2011 |
| 8601 | Message Header Field for Indicating Message Authentication Status | May 2019 |
| 8617 | The Authenticated Received Chain (ARC) Protocol | Jul 2019 |
| 9989 | DMARC | **19 May 2026** |
| 9990 | DMARC Aggregate Reporting | May 2026 |
| 9991 | DMARC Failure Reporting | May 2026 |

They interlock. DMARC alignment is meaningless without SPF and DKIM, and ARC exists only
because forwarding breaks both. A question worth asking usually needs two of them.

**The IETF Trust Legal Provisions** grant the right to distribute IETF documents "in full and
without modification", with no derivative-works right outside the standards process. A chunked
or cleaned copy in a public repository would breach that. So `corpus/fetch.py` downloads the
files, hashes them, and writes `manifest.json`. **You commit the manifest. You never commit the
documents.** `corpus/rfc/` is gitignored and one of the smoke tests checks it.

Your reviewer runs the same script and gets byte-identical files, and your manifest proves
which bytes your run was built on. That is the same discipline as pinning a model to a dated
snapshot, which this week has already asked of you twice.

```bash
python corpus/fetch.py --verify    # re-hash what is on disk, compare to the manifest
```

### One thing to know before you write a question

**RFC 9989 obsoleted RFC 7489 on 19 May 2026.** Every model you can reach was trained on 7489.

So your agent, left to its own knowledge, will produce fluent and confident DMARC claims that
the corpus contradicts. **The corpus is the truth.** If your brief disagrees with it, your
brief is wrong however sure it sounds, and the validator is what catches it: a claim from
memory either cites nothing or cites a ledger entry that does not support it.

---

## 3. The ledger, which is the project

`app/ledger.py` ships `NaiveLedger`. It is Week 5's citation model, ported unchanged, and it is
correct for a single response and wrong for a brief.

Week 5's validator rested on one sentence: *the retrieved chunks define the set of legal IDs.*
True of one query. False of a brief assembled from six rounds.

`NaiveLedger` numbers evidence per query, so the first hit of any query is `doc#1`. Run it and
watch:

```
round 1, doc#1 pointed at : rfc9989:0007 | 4.4 Identifier Alignment
after round 2, doc#1 is   : rfc9990:0031 | 6.2 Reporting
```

A claim written in round 1 citing `doc#1` now resolves to a passage about something else.
**Nothing crashes, nothing logs, and the validator passes it**, because the quote it checks is
round two's quote against round two's chunk. The brief looks correct and cites confidently.

Your job is a ledger with three properties, which are the three tests:

1. **Stable.** An id names the same passage from the moment it is issued until the run ends.
2. **Deduplicated.** One passage found by two queries is one entry with one id. `queries`
   accumulates.
3. **Traceable.** Every entry records the round that first surfaced it and every query that
   returned it.

Keep the `LedgerProtocol` interface, or change the callers too and say so in your README.

---

## 4. The stop condition

The loop inherits four guards from OpsAssist: the iteration cap that bounds it, the wall
clock, the token budget, and the dollar ceiling that is off until you set `MAX_COST_USD`. Every
one of them answers *has this run cost too much?*

None of them answers *is this research finished?* That is a coverage question, and OpsAssist
never had one because an ops task ends when the model says it does.

Add the guard that answers it, give it a `stop_reason` of its own so the trace names it, and
put its settings in `config.py`. **An inherited iteration cap is not a research stop condition**,
and the memo asks you to say why yours is.

---

## 5. Your three questions

`questions.json`, three entries, each declaring what you expect:

- **answerable** - the corpus fully settles it.
- **partial** - the corpus settles some of it. The brief must name the missing half.
- **unanswerable** - it sounds in scope and the corpus does not cover it at all. The brief must
  decline and say what is missing.

The third one carries the most marks and is the one people get wrong. A good unanswerable
question is not off-topic; it is squarely on-topic and outside what a specification is for.
RFCs say what a receiver MUST do with a failing check. They never say how many domains do it,
what any vendor requires, or what it costs.

---

## 6. Running it

```bash
python -m app.main                 # uses questions.json
```

Writes five files into the working directory. These, plus `questions.json`,
`corpus/manifest.json` and your README, are the submission:

| File | What it is |
|---|---|
| `brief.md` | The output. One section per question, every claim carrying ledger ids |
| `brief.json` | The same brief, structured. It is what lets a reviewer re-run validation |
| `ledger.json` | The ledger as the run left it |
| `trace.json` | Every iteration, its tool calls and its new evidence count |
| `validation.json` | Per-claim verdicts and per-section totals |

---

## 7. What is graded

20 marks, 12 to pass. The rubric is on the Requirements page; the short version:

- **Correctness (7).** Citations resolve and quotes are verbatim. The unanswerable question is
  declined and the partial one names its gap.
- **Completeness (6).** The ledger carries stable ids and provenance; the questions match their
  declared shape; the trace and validation agree with the brief.
- **Design (4).** Your stop condition, defended. And the reject-complexity argument.
- **Clarity (3).** README in template order, the demo recording, the AI-assistance section.

**Do not commit:** `.env`, your key, `corpus/rfc/`, `.venv/`, `__pycache__/`. One of the smoke
tests asks git whether the corpus is tracked, so `pytest -q` catches the one that matters.

---

## 8. Where this goes next

The capstone paths **FilingScout** and **PulseBrief** are both this project at scale: an agent
that reads a document set and writes a cited brief on a schedule. The ledger you build here is
the part that carries over unchanged.
