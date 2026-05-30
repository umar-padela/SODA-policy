"""
Local entrypoint — sequential π_low sweep on Modal (one GPU, laptop-safe).

All trials run one after another inside a single GPU container so only one GPU
is needed at a time (avoids parallel-quota failures when launching many trials).

``--detach`` MUST appear before the script path (Modal CLI flag, not function arg):

  modal run --detach modal/modal_sweep_train_low.py \\
      --trials-json "[...]" --run-readme "A4 LR sweep"

Typically invoked automatically by ``scripts/sweep_train_low.py --modal``.
Do not call this directly — use the sweep script instead.

Why --detach before the script?
  Modal CLI parses flags before the script path as its own options.
  ``modal run --detach script.py`` → Modal keeps the job alive after disconnect.
  ``modal run script.py --detach`` → passed to main() as a function arg; Modal
  still stops the ephemeral app when the entrypoint exits, killing the job.
"""

import json

from modal_config import app, run_sweep_train_low_sequential, spawn_modal_function


@app.local_entrypoint()
def main(
    trials_json: str,
    run_readme: str = "",
) -> None:
    trials = json.loads(trials_json)
    spawn_modal_function(
        run_sweep_train_low_sequential,
        label=f"sweep_train_low_sequential:{len(trials)}_trials",
        wait=False,
        trials=trials,
    )
