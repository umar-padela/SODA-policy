"""
Train bottleneck_expert_positive_negative — termination_study stage 1 extension.

250 epochs, frozen backbone, only β MLP trains.
Adds escape relabeling (K-1 wrong-option examples per anchor) to bottleneck_expert.

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/modal_train_stage1_pn.py \\
    --run-readme "termination_study stage1 bottleneck_expert_positive_negative 250ep"

Checkpoint saved to:
  /experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert_positive_negative/
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, train_low  # noqa: E402

OUTPUT_BASE = "/experiments/final_experiments/pusht/termination_study/stage1"
CONFIG_NAME = "exp_term_bottleneck_expert_positive_negative"
LABEL = "bottleneck_expert_positive_negative"


@app.local_entrypoint()
def main(run_readme: str = "") -> None:
    output_dir = f"{OUTPUT_BASE}/{LABEL}"
    hydra_overrides = [
        f"train_low.output_dir={output_dir}",
        f"train_low.wandb_run_name=term_stage1_{LABEL}",
    ]
    call = train_low.spawn(
        config_name=CONFIG_NAME,
        task="pusht",
        hydra_overrides=hydra_overrides,
        run_readme=f"{run_readme} [{LABEL}]",
    )
    print(f"Spawned {LABEL} → {output_dir}")
    call.get()
    print(f"Done → {output_dir}")
    print("Next: run modal_eval_stage1.py with --runs bottleneck_expert_positive_negative")
