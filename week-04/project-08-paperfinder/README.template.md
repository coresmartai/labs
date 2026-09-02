# PaperFinder

<!-- Rename this file to README.md before you open the pull request.
     Keep the headings. The reviewer reads them in this order and marks
     the three sections flagged as graded. Replace every <...> placeholder. -->

Five arXiv papers, one KnowledgeVault index, one benchmark. Built by <your name>
in Week 4 of the CoreSmart Applied GenAI and Agentic AI Engineering program.

## How to run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # set OPENAI_API_KEY; leave QDRANT_MODE=local
# download the five PDFs listed in bench/papers.json into data/sample/ under the exact filenames
uvicorn app.main:app --reload
curl -X POST "http://localhost:8000/ingest/all?reset=true" > bench/ingest_report.json
curl http://localhost:8000/stats > bench/stats.json
python bench/eval.py --set all --label baseline        # after bench/my_queries.jsonl is written
pytest -q
```

## The corpus (graded: R1, R2)

Blocks and the two figure counts come from `bench/ingest_report.json`; prose chunks from `bench/stats.json`.

| document_id | title | blocks | prose chunks | figures described | figures dropped |
|---|---|---|---|---|---|
| attention-is-all-you-need | Attention Is All You Need | <n> | <n> | <n> | <n> |
| bert | BERT | <n> | <n> | <n> | <n> |
| rag | Retrieval-Augmented Generation | <n> | <n> | <n> | <n> |
| vit | An Image is Worth 16x16 Words | <n> | <n> | <n> | <n> |
| lora | LoRA | <n> | <n> | <n> | <n> |

Papers that yielded zero figures, and why: <name them, and say what the
parser needs before it would see their figures. One paragraph.>

## Baseline numbers (graded: R5)

From `bench/results_baseline.json`, k = <k>, widen = <true/false>:

| set | queries | scored | unreachable | recall@k | prose | figure | MRR | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| given | 10 | <n> | <n> | <x.xx> | <x.xx> | <x.xx> | <x.xx> | <n> | <n> |
| mine | <n> | <n> | <n> | <x.xx> | <x.xx> | <x.xx> | <x.xx> | <n> | <n> |

## My golden set (graded: R4)

<Three or four sentences. What kinds of question you chose to cover and
why. Which paper or modality you deliberately made hard. One query you
expected to hit and it did not, and what you learned from the miss.>

## The one change (graded: R6)

**What I changed:** <one sentence. Exactly one thing.>

**Why I expected it to help:** <two or three sentences, written before you
ran it. Which misses in the baseline pointed at this change?>

**Before and after** (`bench/results_baseline.json` against `bench/results_<label>.json`):

| | recall@k given | recall@k mine | figure recall | p50 ms | p95 ms |
|---|---|---|---|---|---|
| before | <x.xx> | <x.xx> | <x.xx> | <n> | <n> |
| after | <x.xx> | <x.xx> | <x.xx> | <n> | <n> |

**What actually happened, and would I ship it:** <a short paragraph. If it
did not help, say so and say why; a negative result with a reason scores
the same as a positive one. If it helped one set and hurt the other,
that is the interesting case, so explain it.>

## Model pins

| purpose | model |
|---|---|
| figure describer | gpt-5.4-mini-2026-03-17 |
| fast re-describe | gpt-5.4-nano-2026-03-17 |
| embeddings | text-embedding-3-large (3,072 dims) |

<If you changed any of these, say which and why.>
