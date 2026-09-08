# results/

Two files go here, both from your final run and both committed:

- `run.json`: the per-case run, written by `POST /run` with
  `"save_results_to": "../results/run.json"`.
- `human_scores.json`: your ten blind scores, `{case_id: score}`, written by
  hand after `python scripts/human_slice.py show ../results/run.json`.

This folder is excluded from the repository leakage scan on purpose: the run
file is the evaluation record and necessarily carries every prompt. Nothing
else in the repository may carry seed text. Delete this file if you like.
