# Responsible AI checklist: SafeAssist

TASK 6. Rename this file to `RESPONSIBLE_AI.md` and complete it.

One rule, and the test enforces it: **a ticked item must name an artefact in
this repository, in backticks, and that file must exist.** An item you cannot
attach a file to is an intention, not a control, and it stays unticked.

Leaving items unticked is expected and is not a lost mark. A checklist where
everything is ticked after one afternoon was filled in rather than run.

## 1. Privacy controls

- [ ] A documented detector configuration, including the entity allowlist and every score changed from its default: `path/to/file`
- [ ] The domain entity this deployment holds, registered as a recogniser: `path/to/file`
- [ ] Access rules enforced at retrieval time rather than after it: `path/to/file`
- [ ] An append-only audit record carrying no values: `path/to/file`
- [ ] Operational telemetry scrubbed at the call site: `path/to/file`

## 2. Output monitoring

- [ ] A labelled evaluation set with negatives and hard cases: `path/to/file`
- [ ] A scorecard reporting precision and recall per entity: `path/to/file`
- [ ] Alerting on audit-event rates in a running deployment

## 3. Human in the loop

- [ ] The low-confidence band creates a record a person can act on: `path/to/file`
- [ ] Decisions are terminal and append-only: `path/to/file`
- [ ] A timeout, so the queue cannot become a silent backlog: `path/to/file`
- [ ] A reviewer interface

## 4. Bias assessment

- [ ] Detection recall measured per demographic group

## 5. Communications plan

- [ ] What a user is told when their input is redacted, and when

---

For every item you leave unticked, write one sentence saying why. "Not done"
with a reason is a finding. "Not done" with no reason is an omission.
