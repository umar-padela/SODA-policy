"""
Train the `both` combination — termination_study stage 2.

Stage 2 trains only the `both` run (concat of bottleneck_best + obs_best features).
bottleneck_best and obs_best are reused from stage 1a and 1b — no retraining needed.

Before running: update exp_term_both.yaml with the stage 1b winner's escape settings
(escape_relabeling, escape_prob), the same way kernel_size was updated after the
kernel sweep.

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage2_input_comparison/modal_train_stage2.py \\
    --run-readme "termination_study stage2 both combination"

Checkpoint saved to:
  /experiments/final_experiments/pusht/termination_study/stage2/both/
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, train_low  # noqa: E402

OUTPUT_BASE = "/experiments/final_experiments/pusht/termination_study/stage2"


@app.local_entrypoint()
def main(
    run_readme: str = "",
) -> None:
    output_dir = f"{OUTPUT_BASE}/both"
    hydra_overrides = [
        f"train_low.output_dir={output_dir}",
        "train_low.wandb_run_name=term_stage2_both",
    ]
    call = train_low.spawn(
        config_name="exp_term_both",
        task="pusht",
        hydra_overrides=hydra_overrides,
        run_readme=f"{run_readme} [both]",
    )
    print(f"Spawned both → {output_dir}")
    call.get()
    print(f"  both done → {output_dir}")
    print("\nAll done. Next: run modal_eval_stage2.py")
