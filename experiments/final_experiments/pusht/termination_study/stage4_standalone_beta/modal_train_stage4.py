"""
Train standalone β network — termination_study stage 4.

Architecture: fresh ResNet18Conv (ImageNet init, fully trainable) + option embed + MLP.
Training signal: completion + escape relabeling (positive_negative expand-all).
Decoupled from diffusion: k=9/450-epoch LowPolicy used for obs normalizer only.

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage4_standalone_beta/modal_train_stage4.py \\
    --run-readme "termination_study stage4 standalone ResNet beta"

Checkpoints saved to:
  /experiments/final_experiments/pusht/termination_study/stage4/obs_positive_negative_separate/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, train_beta  # noqa: E402

OUTPUT_DIR = "/experiments/final_experiments/pusht/termination_study/stage4/obs_positive_negative_separate"
CONFIG_NAME = "obs_positive_negative_standalone"


@app.local_entrypoint()
def main(run_readme: str = "") -> None:
    """Train standalone β network (ResNet + MLP, ImageNet init, positive_negative signal)."""
    hydra_overrides = [
        f"train_beta.output_dir={OUTPUT_DIR}",
        "train_beta.wandb_run_name=obs_positive_negative_separate",
    ]

    print(f"Launching train_beta: {CONFIG_NAME}")
    print(f"  output_dir: {OUTPUT_DIR}")
    print(f"  overrides:  {hydra_overrides}")
    print("Safe to close terminal with --detach; job continues on Modal.")

    call = train_beta.spawn(
        config_name=CONFIG_NAME,
        task="pusht",
        hydra_overrides=hydra_overrides,
        run_readme=run_readme or "termination_study stage4 standalone ResNet beta",
    )
    call.get()
    print(f"\nDone. Checkpoints → {OUTPUT_DIR}")
    print("Next: run modal_eval_stage4.py after training completes.")
