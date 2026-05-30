"""
Train the `both` combination — termination_study stage 2.

Stage 2 trains only the `both` run (concat of bottleneck_best + obs_best features).
bottleneck_best and obs_best are reused from stage 1a and 1b — no retraining needed.

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage2_input_comparison/modal_train_stage2.py \\
    --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt \\
    --bottleneck-winner-config exp_term_bottleneck_{expert|ddim5} \\
    --obs-escape True|False \\
    --run-readme "termination_study stage2 both combination"

bottleneck-winner-config: winner from stage 1a (exp_term_bottleneck_expert or exp_term_bottleneck_ddim_positive)
obs-escape: True if obs_positive_negative won stage 1b, False if obs_positive won

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
    base_checkpoint: str,
    run_readme: str,
    bottleneck_winner_config: str = "exp_term_bottleneck_expert",
    obs_escape: bool = False,
) -> None:
    """
    Train the `both` combination (bottleneck_best + obs_best concatenated).

    Parameters
    ----------
    base_checkpoint
        Best checkpoint from kernel_size_study.
    run_readme
        Short description.
    bottleneck_winner_config
        Config name for stage 1a winner (exp_term_bottleneck_expert or exp_term_bottleneck_ddim_positive).
    obs_escape
        True if obs_positive_negative won stage 1b; sets escape_relabeling accordingly.
    """
    output_dir = f"{OUTPUT_BASE}/both"
    hydra_overrides = [
        f"train_low.output_dir={output_dir}",
        f"train_low.checkpoint={base_checkpoint}",
        "train_low.wandb_run_name=term_stage2_both",
        # Propagate the bottleneck source from stage 1a winner
        f"low_policy.termination_input=both",
        # Propagate escape signal from stage 1b winner
        f"low_policy.escape_relabeling={str(obs_escape).lower()}",
        f"low_policy.escape_prob={'0.5' if obs_escape else '0.0'}",
    ]
    call = train_low.spawn(
        config_name="exp_term_both",
        task="pusht",
        hydra_overrides=hydra_overrides,
        run_readme=f"{run_readme} [both]",
    )
    print(f"Spawned both → {output_dir}")
    print(f"  bottleneck source: {bottleneck_winner_config}")
    print(f"  obs escape relabeling: {obs_escape}")

    call.get()
    print(f"  both done → {output_dir}")
    print("\nAll done. Next: run modal_eval_stage2.py")
