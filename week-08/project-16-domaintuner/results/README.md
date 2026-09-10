# Your results go here

Two files, both emitted by the evaluation harness rather than written by hand.

- `base.json` from the run with no adapter
- `tuned.json` from the run with yours

Each is one `EvalResult` row with eight fields: approach, model id, accuracy, p50 and p95 latency,
cost per thousand calls, trainable parameters, and privacy posture.

**Every number in your memo has to appear in one of these two files.** A figure that exists only in
prose scores zero on that rubric line, and it is the fastest two marks on this brief to lose.

Do not edit them by hand. If a number looks wrong, re-run rather than correcting the file.
