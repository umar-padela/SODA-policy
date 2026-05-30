"""
Train k=5, k=7, and k=9 policies (no β) — kernel_size_study step 1.

Runs are launched in parallel (separate Modal GPU containers).
Checkpoints saved to:
  /experiments/final_experiments/pusht/kernel_size_study/k5/
  /experiments/final_experiments/pusht/kernel_size_study/k7/
  /experiments/final_experiments/pusht/kernel_size_study/k9/

Usage (repo root):
  # All three at once
  modal run --detach experiments/final_experiments/pusht/kernel_size_study/modal_train_kernels.py \\
    --run-readme "kernel_size_study k=5 k=7 k=9 no-beta 500ep"

  # k=5 only first (sanity check before committing to k=7 and k=9)
  modal run --detach experiments/final_experiments/pusht/kernel_size_study/modal_train_kernels.py \\
    --kernels k5 --run-readme "kernel_size_study k=5 no-beta 500ep"

  # k=7 and k=9 after k=5 looks good
  modal run --detach experiments/final_experiments/pusht/kernel_size_study/modal_train_kernels.py \\
    --kernels k7,k9 --run-readme "kernel_size_study k=7 k=9 no-beta 500ep"
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))

from modal_config import app, train_low  # noqa: E402

OUTPUT_BASE = "/experiments/final_experiments/pusht/kernel_size_study"

ALL_CONFIGS = {
    "k5": ("exp_k5_no_beta", "k5_no_beta_500ep"),
    "k7": ("exp_k7_no_beta", "k7_no_beta_500ep"),
    "k9": ("exp_k9_no_beta", "k9_no_beta_500ep"),
}


@app.local_entrypoint()
def main(run_readme: str, kernels: str = "k5,k7,k9") -> None:
    """
    Train selected kernels in parallel.

    Parameters
    ----------
    run_readme
        Short description archived with each run.
    kernels
        Comma-separated subset of {k5, k7, k9} to train (default: all three).
        Example: --kernels k5   or   --kernels k7,k9
    """
    selected = [k.strip() for k in kernels.split(",")]
    unknown = [k for k in selected if k not in ALL_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown kernels: {unknown}. Choose from {list(ALL_CONFIGS)}")

    calls = []
    for k_label in selected:
        config_name, run_name = ALL_CONFIGS[k_label]
        output_dir = f"{OUTPUT_BASE}/{k_label}"
        hydra_overrides = [
            f"train_low.output_dir={output_dir}",
            f"train_low.wandb_run_name={run_name}",
        ]
        call = train_low.spawn(
            config_name=config_name,
            task="pusht",
            hydra_overrides=hydra_overrides,
            run_readme=f"{run_readme} [{k_label}]",
        )
        calls.append((k_label, output_dir, call))
        print(f"Spawned {k_label} training → {output_dir}")

    print(f"\nWaiting for {len(calls)} training run(s) to complete...")
    for k_label, output_dir, call in calls:
        call.get()
        print(f"  {k_label} done → {output_dir}")

    print("\nAll done. Next: run modal_eval_kernel_sweep.py (n_action_steps=8 fixed)")
