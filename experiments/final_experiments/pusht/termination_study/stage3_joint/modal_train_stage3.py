"""
Joint training (stop_grad=False) for all 5 stage 3 runs — termination_study stage 3.

Repeats stage 1a, 1b, and stage 2 with β gradients flowing into the backbone (λ=0.01).
Run stage 1a+1b joints first (4 in parallel), then both_joint after picking winners.

Stage 3 runs (5 total):
  From stage 1a: bottleneck_expert_joint, bottleneck_ddim_positive_joint
  From stage 1b: obs_positive_joint, obs_positive_negative_joint
  From stage 2:  both_joint  (run after 1a+1b decisions)

Usage (repo root):
  # Step 1: run stage 1a+1b joints in parallel (4 runs)
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage3_joint/modal_train_stage3.py \\
    --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt \\
    --group stage1 \\
    --run-readme "termination_study stage3 joint stage1 runs"

  # Step 2: after picking winners, run both_joint (1 run)
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage3_joint/modal_train_stage3.py \\
    --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt \\
    --group stage2 \\
    --obs-escape True|False \\
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
STAGE1_RUNS = [
    ("exp_term_bottleneck_expert",                  "bottleneck_expert_joint",                  []),
    ("exp_term_bottleneck_ddim_positive",           "bottleneck_ddim_positive_joint",           []),
    ("exp_term_bottleneck_ddim_positive_negative",  "bottleneck_ddim_positive_negative_joint",  []),
    ("exp_term_obs_positive",                       "obs_positive_joint",                       []),
    ("exp_term_obs_positive_negative",              "obs_positive_negative_joint",              []),
]


@app.local_entrypoint()
def main(
    base_checkpoint: str,
    run_readme: str,
    group: str = "stage1",
    obs_escape: bool = False,
) -> None:
    """
    Parameters
    ----------
    base_checkpoint
        Best checkpoint from kernel_size_study.
    run_readme
        Short description.
    group
        "stage1" to run the 4 joint repeats of 1a+1b in parallel.
        "stage2" to run both_joint after 1a+1b winners are known.
    obs_escape
        (stage2 only) True if obs_positive_negative won stage 1b.
    """
    if group == "stage1":
        runs = STAGE1_RUNS
    elif group == "stage2":
        escape_overrides = (
            ["low_policy.escape_relabeling=true", "low_policy.escape_prob=0.5"]
            if obs_escape else
            ["low_policy.escape_relabeling=false"]
        )
        runs = [("exp_term_both_joint", "both_joint", escape_overrides)]
    else:
        raise ValueError(f"group must be 'stage1' or 'stage2', got {group!r}")

    calls = []
    for config_name, label, extra_overrides in runs:
        output_dir = f"{OUTPUT_BASE}/{label}"
        hydra_overrides = [
            f"train_low.output_dir={output_dir}",
            f"train_low.checkpoint={base_checkpoint}",
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
