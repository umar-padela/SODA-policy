"""
Train bottleneck_expert_positive_negative_separate — termination_study stage 1 extension.

Fresh UNet (random init) + ResNet (ImageNet init), gradient flows end-to-end from BCE
through β MLP → UNet bottleneck → ResNet + option_embed. No diffusion MSE loss.

250 epochs, 5-epoch linear warmup, cosine decay.

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/modal_train_stage1_separate.py \\
    --run-readme "stage1 bottleneck_expert_positive_negative_separate 250ep end-to-end"

Checkpoint saved to:
  /experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert_positive_negative_separate/
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, train_low  # noqa: E402

OUTPUT_BASE = "/experiments/final_experiments/pusht/termination_study/stage1"
CONFIG_NAME = "exp_term_bottleneck_expert_positive_negative_separate"
LABEL = "bottleneck_expert_positive_negative_separate"


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
    print("Next: run modal_eval_stage1_separate.py")
