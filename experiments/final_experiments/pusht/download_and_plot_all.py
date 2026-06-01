"""
Download all study results from Modal Volume and regenerate all plots.

Usage:
  python experiments/final_experiments/pusht/download_and_plot_all.py

Each study's --download-only modal script fetches the JSON from the volume
and re-runs the plot script automatically. The comparison study has no
--download-only flag so it uses modal volume get + a direct plot call.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  WARNING: exited with code {result.returncode}")


STUDIES = [
    (
        "Kernel size study (download + plot)",
        ["modal", "run",
         str(HERE / "kernel_size_study/modal_eval_kernel_sweep.py"),
         "--download-only"],
    ),
    (
        "Receding horizon study (download + plot)",
        ["modal", "run",
         str(HERE / "receding_horizon_study/modal_eval_n_action_steps.py"),
         "--download-only"],
    ),
    (
        "Termination study stages 1 + 1b (download + plot)",
        ["modal", "run",
         str(HERE / "termination_study/modal_eval_stage1_all.py"),
         "--download-only"],
    ),
    (
        "Termination study stage 2 (download + plot)",
        ["modal", "run",
         str(HERE / "termination_study/stage2_input_comparison/modal_eval_stage2.py"),
         "--download-only"],
    ),
    (
        "Termination study stage 3 (download + plot)",
        ["modal", "run",
         str(HERE / "termination_study/stage3_joint/modal_eval_stage3.py"),
         "--download-only"],
    ),
]

COMPARISON_LOCAL = HERE / "comparison_study/final_comparison.json"
COMPARISON_VOLUME = "final_experiments/pusht/comparison_study/final_comparison.json"
COMPARISON_PLOT = HERE / "comparison_study/plot_comparison.py"


def main() -> None:
    for label, cmd in STUDIES:
        run(cmd, label)

    run(
        ["modal", "volume", "get", "soda-experiments",
         COMPARISON_VOLUME, str(COMPARISON_LOCAL)],
        "Comparison study (download JSON)",
    )
    run(
        [sys.executable, str(COMPARISON_PLOT),
         "--data", str(COMPARISON_LOCAL)],
        "Comparison study (plot)",
    )

    print(f"\n{'=' * 60}")
    print("  All done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
