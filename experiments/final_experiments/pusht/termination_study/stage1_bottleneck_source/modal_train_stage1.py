"""
Train bottleneck_expert and bottleneck_ddim_positive in parallel — termination_study stage 1.

Both runs: 100 epochs, frozen backbone, only β MLP trains.
Warmstart checkpoint is set in each config file (exp_term_bottleneck_*.yaml).

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/modal_train_stage1.py \\
    --run-readme "termination_study stage1 bottleneck source comparison"

Checkpoints saved to:
  /experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert/
  /experiments/final_experiments/pusht/termination_study/stage1/bottleneck_ddim_positive/
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, train_low  # noqa: E402

OUTPUT_BASE = "/experiments/final_experiments/pusht/termination_study/stage1"


ALL_RUNS = [
    ("exp_term_bottleneck_expert", "bottleneck_expert"),
    ("exp_term_bottleneck_ddim_positive", "bottleneck_ddim_positive"),
]


@app.local_entrypoint()
def main(
    run_readme: str = "",
    runs: str = "",
) -> None:
    """
    Train bottleneck_expert and bottleneck_ddim_positive (in parallel by default).

    Warmstart checkpoint is read from each config file — no CLI override needed.
    """
    selected = {r.strip() for r in runs.split(",") if r.strip()} if runs else None
    configs = [
        (cfg, label) for cfg, label in ALL_RUNS
        if selected is None or label in selected
    ]
    if not configs:
        raise ValueError(f"No matching runs. Available: {[l for _, l in ALL_RUNS]}")

    calls = []
    for config_name, label in configs:
        output_dir = f"{OUTPUT_BASE}/{label}"
        hydra_overrides = [
            f"train_low.output_dir={output_dir}",
            f"train_low.wandb_run_name=term_stage1_{label}",
        ]
        call = train_low.spawn(
            config_name=config_name,
            task="pusht",
            hydra_overrides=hydra_overrides,
            run_readme=f"{run_readme} [{label}]",
        )
        calls.append((label, output_dir, call))
        print(f"Spawned {label} → {output_dir}")

    print(f"\nWaiting for {len(calls)} training runs...")
    for label, output_dir, call in calls:
        call.get()
        print(f"  {label} done → {output_dir}")

    print("\nAll done. Next: run modal_eval_stage1.py")
