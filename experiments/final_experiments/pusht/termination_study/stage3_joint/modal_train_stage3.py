"""
Joint training (stop_grad=False) for all 5 stage 3 runs — termination_study stage 3.

Repeats stage 1a, 1b, and stage 2 with β gradients flowing into the backbone (λ=0.01).
Run stage 1a+1b joints first (in parallel), then both_joint after picking winners.

Stage 3 runs (5 total):
  From stage 1a: bottleneck_expert_joint, bottleneck_ddim_positive_joint
  From stage 1b: obs_positive_joint, obs_positive_negative_joint
  From stage 2:  both_joint  (run after 1a+1b decisions)

Before running stage2 group: update exp_term_both_joint.yaml with the stage 1b winner's
escape settings (escape_relabeling, escape_prob), the same way kernel_size was updated
after the kernel sweep.

Usage (repo root):
  # Step 1: run stage 1a+1b joints in parallel
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage3_joint/modal_train_stage3.py \\
    --group stage1 \\
    --run-readme "termination_study stage3 joint stage1 runs"

  # Step 2: update exp_term_both_joint.yaml, then run both_joint
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage3_joint/modal_train_stage3.py \\
    --group stage2 \\
    --run-readme "termination_study stage3 joint both"

Checkpoints:
  /experiments/final_experiments/pusht/termination_study/stage3/bottleneck_expert_joint/
  /experiments/final_experiments/pusht/termination_study/stage3/bottleneck_ddim_positive_joint/
  /experiments/final_experiments/pusht/termination_study/stage3/obs_positive_joint/
  /experiments/final_experiments/pusht/termination_study/stage3/obs_positive_negative_joint/
  /experiments/final_experiments/pusht/termination_study/stage3/both_joint/
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, train_low  # noqa: E402

OUTPUT_BASE = "/experiments/final_experiments/pusht/termination_study/stage3"

# stage1 group: repeat 1a and 1b with joint training (5 runs in parallel)
# Uses dedicated joint configs (stop_grad=False, skip_diffusion_loss=False, λ=0.01)
# instead of the frozen stage1 configs.
STAGE1_RUNS = [
    ("exp_term_bottleneck_expert_joint",                  "bottleneck_expert_joint",                  []),
    ("exp_term_bottleneck_ddim_positive_joint",           "bottleneck_ddim_positive_joint",           []),
    ("exp_term_bottleneck_ddim_positive_negative_joint",  "bottleneck_ddim_positive_negative_joint",  []),
    ("exp_term_obs_positive_joint",                       "obs_positive_joint",                       []),
    ("exp_term_obs_positive_negative_joint",              "obs_positive_negative_joint",              []),
]


@app.local_entrypoint()
def main(
    run_readme: str = "",
    group: str = "stage1",
) -> None:
    """
    Parameters
    ----------
    run_readme
        Short description.
    group
        "stage1" to run joint repeats of 1a+1b in parallel.
        "stage2" to run both_joint (update exp_term_both_joint.yaml first).

    Warmstart checkpoint is set in each config file — no CLI override needed.
    """
    if group == "stage1":
        runs = STAGE1_RUNS
    elif group == "stage2":
        runs = [("exp_term_both_joint", "both_joint", [])]
    else:
        raise ValueError(f"group must be 'stage1' or 'stage2', got {group!r}")

    calls = []
    for config_name, label, extra_overrides in runs:
        output_dir = f"{OUTPUT_BASE}/{label}"
        hydra_overrides = [
            f"train_low.output_dir={output_dir}",
            f"train_low.wandb_run_name=term_stage3_{label}",
            *extra_overrides,
        ]
        call = train_low.spawn(
            config_name=config_name,
            task="pusht",
            hydra_overrides=hydra_overrides,
            run_readme=f"{run_readme} [{label}]",
        )
        calls.append((label, output_dir, call))
        print(f"Spawned {label} → {output_dir}")

    print(f"\nWaiting for {len(calls)} joint training run(s)...")
    for label, output_dir, call in calls:
        call.get()
        print(f"  {label} done → {output_dir}")

    print("\nAll done. Next: run modal_eval_stage3.py")
