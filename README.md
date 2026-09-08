# labs

Public course and lab code for CoreSmart.AI cohorts.

This repository holds the reference notebooks and starter code that CoreSmart's
17-Week Applied GenAI & Agentic AI Engineering program links to directly. It is
public on purpose: learners open these files in Google Colab straight from
GitHub, so nothing here should ever contain a secret. API keys live in a `.env`
file in the learner's own repository and are never committed anywhere.

## What is here

One folder per week. Each week holds the guided reference lab, which learners
read and run, and the starter for that week's graded project, which they copy
out into their own repository.

```
labs/
  prep-week/     W00-LAB environment check notebook, opened in Colab
  week-01/       releasebot (guided)        project-02-minutemaker (graded)
  week-02/       intentiq (guided)          project-04-feedbacksorter (graded)
  week-03/       orderbot, ticketstream     project-06-reviewrouter (graded)
  week-04/       knowledgevault (guided)    project-08-paperfinder (graded)
  week-05/       citationrag (guided)       project-10-docurag (graded)
  week-06/       ragoptimizer (guided)      project-12-optimizerag (graded)
  week-07/       breakrag (guided)          project-14-ragbench (graded)
```

Each guided folder follows the same shape: `app/` for the service, `tests/` for
the pytest suite, a notebook, `requirements.txt`, `.env.example`, `.gitignore`
and its own `README.md`. Each graded starter holds the data the brief needs and
a README for you to fill in. Its exact filename differs between weeks, so take
it from that week's brief rather than from this listing.

The exact copy commands for every week live in that week's brief in the LMS.
Use those rather than guessing paths from this listing.

## How learners use it

They do not build inside this repository. Their graded work lives in their own
repository, named `genai-17`, which they create once in prep week by pressing
**Use this template** on
[coresmartai/genai-17-template](https://github.com/coresmartai/genai-17-template).
It belongs to their own GitHub account, not to a CoreSmart organisation, and
there is no invitation to accept.

This repository is read-only reference. Learners open the notebooks here in
Colab and follow along, then copy the guided code and the graded starter out
into their own repository and build their version there.

The links that point here are baked into the LMS pages, so the folder and file
names above must not be renamed without updating those pages.

## Keys, and why the warning is stronger than it used to be

**A learner's `genai-17` repository is public.** That is deliberate: it is what
they point an employer at when the course ends, and it is how their work is
read without anyone granting access by hand. It also changes what a leaked key
costs.

A key that reaches a public commit is found by scanning bots within minutes,
and deleting it in a later commit does not help, because it stays in the
history. If you think you have pushed a key, **revoke it at the provider
first**, then ask for help cleaning the history. In that order. Revoking is
something you can do alone in under a minute; rewriting history is not, and the
key is being used while you wait.

Every `.env.example` in this repository is safe to commit. No `.env` ever is.
Both this repository and the template already list `.env` in `.gitignore`.

## For the CoreSmart team

If you rename this repository or move a file, the Colab links on the W00-LAB
(Environment Check) and W01-LAB (ReleaseBot) LMS pages break. Do not hand-edit
those links. Change the `REPO` / `NB_PATH` constant in the matching build script
and re-run it, then re-upload the regenerated page.

Every graded brief hard-codes its own `week-NN/project-NN-name/` copy paths, so
renaming a project folder breaks that week's scaffold step. The org name this
repository lives under is `coresmartai`.

**Two cohorts read this repository at once.** Content under a week that a live
cohort has not yet reached can be changed; content under a week they are
working through should not be, because a mid-flight change to a graded starter
lands under people already building against it.

## Licence and use

Reference material for enrolled CoreSmart cohorts. Contact
training@coresmart.ai with questions.
