# labs

Public course and lab code for CoreSmart.AI cohorts.

This repository holds the reference notebooks and starter code that CoreSmart's
17-Week Applied GenAI & Agentic AI Engineering program links to directly. It is
public on purpose: students open these files in Google Colab straight from
GitHub, so nothing here should ever contain a secret. API keys live in each
student's own private repo, in a `.env` file that is never committed.

## What is here

```
labs/
  prep-week/
    W00-LAB_environment-check.ipynb   Prep-week environment check. The LMS
                                      "Environment Check" item opens this in
                                      Colab. It confirms Python, pip and a few
                                      core libraries are working before Week 1.
  week-01/
    releasebot/                       The Week 1 ReleaseBot reference lab: a
                                      small FastAPI service with an LLM client,
                                      tests, and two notebooks.
      app/                            The service (main, config, llm, tools,
                                      schemas).
      tests/                          pytest suite (test_endpoint.py).
      week1_notebook.ipynb            Exercises every endpoint, cell by cell.
      releasebot_colab.ipynb          Full Colab walk-through; clones this repo.
      requirements.txt                Dependencies.
      .env.example                    Copy to .env and add your own key. The
                                      real .env is git-ignored and never shared.
      README.md                       ReleaseBot's own setup and run notes.
```

## How students use it

They do not clone or fork this repo for their own work. Their graded work lives
in their own private cohort repo (`yourname-genai-17`). This repository is
read-only reference: they open a notebook here in Colab, follow along, and build
their own version in their own repo. The links that point here are baked into
the LMS pages, so the folder and file names above must not be renamed without
updating those pages.

## For the CoreSmart team

If you rename this repo or move a file, the Colab links on the W00-LAB
(Environment Check) and W01-LAB (ReleaseBot) LMS pages break. Do not hand-edit
those links. Change the `REPO` / `NB_PATH` constant in the matching build script
and re-run it, then re-upload the regenerated page. The org name this repo lives
under is `coresmartai`.

## Licence and use

Reference material for enrolled CoreSmart cohorts. Contact
training@coresmart.ai with questions.
