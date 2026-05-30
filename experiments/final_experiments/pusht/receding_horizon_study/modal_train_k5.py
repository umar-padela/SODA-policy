"""
OBSOLETE — k=5 training is now done in kernel_size_study/modal_train_k7_k9.py alongside
k=7 and k=9. The checkpoint lands at:
  /experiments/final_experiments/pusht/kernel_size_study/k5/

Do not run this script. It is kept for reference only.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make modal/ importable from experiments/
sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))

from modal_config import app, spawn_modal_function, train_low  # noqa: E402

# Output directory on Modal Volume
OUTPUT_DIR = "/experiments/final_experiments/pusht/receding_horizon_study/k5"


@app.local_entrypoint()
def main(run_readme: str) -> None:
    """
    Train k=5, no β, 500 epochs, constant LR.

    Parameters
    ----------
    run_readme
        Short description archived with the run (required by Modal infra).
    """
    hydra_overrides = [
        f"train_low.output_dir={OUTPUT_DIR}",
        "train_low.wandb_run_name=k5_no_beta_500ep",
    ]
    spawn_modal_function(
        train_low,
        label="train_low:receding_horizon_study/k5",
        wait=True,
        config_name="exp_k5_no_beta",
        task="pusht",
        hydra_overrides=hydra_overrides,
        run_readme=run_readme,
    )
    print(f"\nCheckpoint saved to: {OUTPUT_DIR}")
    print("Next step: run modal_eval_n_action_steps.py with --low-checkpoint path")
