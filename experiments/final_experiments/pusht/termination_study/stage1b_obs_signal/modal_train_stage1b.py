"""
Train obs_positive and obs_positive_negative in parallel — termination_study stage 1b.

Both runs: 100 epochs, frozen backbone, only β MLP trains.
Starts from best kernel_size checkpoint (set --base-checkpoint).

obs_positive      — completion signal only (β=1 at last frame of each segment)
obs_positive_negative — completion + escape signal (β=1 at last frame AND when
                      option_id replaced with wrong option, p=0.5)

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/modal_train_stage1b.py \\
    --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt \\
    --run-readme "termination_study stage1b obs signal comparison"

Checkpoints saved to:
  /experiments/final_experiments/pusht/termination_study/stage1b/obs_positive/
  /experiments/final_experiments/pusht/termination_study/stage1b/obs_positive_negative/

NOTE: escape_relabeling requires OptionAwareDataset to support escape_prob.
See experimental_plan.md §4 Stage 1b for implementation details.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, train_low  # noqa: E402

OUTPUT_BASE = "/experiments/final_experiments/pusht/termination_study/stage1b"


ALL_RUNS = [
    ("exp_term_obs_positive", "obs_positive"),
    ("exp_term_obs_positive_negative", "obs_positive_negative"),
    ("exp_term_bottleneck_ddim_positive_negative", "bottleneck_ddim_positive_negative"),
]


@app.local_entrypoint()
def main(base_checkpoint: str, run_readme: str, runs: str = "") -> None:
    """
    Train stage 1b runs (in parallel by default).

    Parameters
    ----------
    base_checkpoint
        Volume path to the best checkpoint from kernel_size_study.
    run_readme
        Short description archived with each run.
    runs
        Comma-separated subset to train, e.g. --runs obs_positive.
        Omit to train all runs in parallel.
    """
    selected = {r.strip() for r in runs.split(",") if r.strip()} if runs else None
    configs = [
        (cfg, label) for cfg, label in ALL_RUNS
        if selected is None or label in selected
    ]
    if not configs:
        raise ValueError(f"No matching runs. Available: {[l for _, l in ALL_RUNS]}")

    calls = []
    for config_name, label in configs:
        output_dir = f"{OUTPUT_BASE}/{label}"
        hydra_overrides = [
            f"train_low.output_dir={output_dir}",
            f"train_low.checkpoint={base_checkpoint}",
            f"train_low.wandb_run_name=term_stage1b_{label}",
        ]
        call = train_low.spawn(
            config_name=config_name,
            task="pusht",
            hydra_overrides=hydra_overrides,
            run_readme=f"{run_readme} [{label}]",
        )
        calls.append((label, output_dir, call))
        print(f"Spawned {label} → {output_dir}")

    print(f"\nWaiting for {len(calls)} training runs...")
    for label, output_dir, call in calls:
        call.get()
        print(f"  {label} done → {output_dir}")

    print("\nAll done. Next: run modal_eval_stage1b.py")
