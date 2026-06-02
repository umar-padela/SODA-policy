"""
Resume bottleneck_expert from epoch 100 → 250 — termination_study stage 1 extension.

Restores optimizer, LR scheduler, and EMA states from epoch_0100.ckpt.
Logs to the same W&B run as the original training (pass --wandb-resume-id).

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/modal_train_stage1_resume.py \\
    --wandb-resume-id <RUN_ID>

  <RUN_ID>: find in W&B UI → open the bottleneck_expert run → copy from URL:
    https://wandb.ai/<entity>/<project>/runs/<RUN_ID>

Checkpoints saved to (same dir as original run, epoch_0110 … epoch_0250):
  /experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert/
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, train_low  # noqa: E402

OUTPUT_DIR = "/experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert"


@app.local_entrypoint()
def main(
    wandb_resume_id: str = "",
    run_readme: str = "",
) -> None:
    if not wandb_resume_id:
        print(
            "WARNING: --wandb-resume-id not set. Training will resume but log to a NEW W&B run.\n"
            "Find the run ID in the W&B UI URL and re-run with --wandb-resume-id <id> "
            "to continue the existing curve."
        )

    hydra_overrides = [
        f"train_low.output_dir={OUTPUT_DIR}",
        f"train_low.wandb_run_name=term_stage1_bottleneck_expert_resumed",
    ]
    if wandb_resume_id:
        hydra_overrides.append(f"train_low.wandb_resume_id={wandb_resume_id}")

    call = train_low.spawn(
        config_name="exp_term_bottleneck_expert_resume",
        task="pusht",
        hydra_overrides=hydra_overrides,
        run_readme=f"{run_readme} [bottleneck_expert resume ep101-250]",
    )
    print(f"Spawned resume → {OUTPUT_DIR}")
    call.get()
    print(f"Done. Checkpoints at {OUTPUT_DIR}/epoch_01{{10..50}}.ckpt")
    print("Next: run modal_eval_stage1.py (it will pick up the new epoch checkpoints)")
