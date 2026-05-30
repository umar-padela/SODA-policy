"""
Plot kernel size study results — kernel_size_study step 3.

Outputs (all in the same dir as --data):
  plot_kernel_vs_epoch_t300.png   — mean max overlap vs epoch, t≤300 (main metric)
  plot_kernel_vs_epoch_t250.png   — t≤250
  plot_kernel_vs_epoch_t200.png   — t≤200
  plot_kernel_vs_epoch_t150.png   — t≤150
  plot_kernel_vs_epoch_grid.png   — all horizons in a subplot grid (t=100–300)
  plot_kernel_best.png            — bar chart: best epoch per kernel (t≤300)

Usage:
  python experiments/final_experiments/pusht/kernel_size_study/plot_kernel_study.py \\
    --data experiments/final_experiments/pusht/kernel_size_study/kernel_sweep_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "shared"))
from experiment_utils import (  # noqa: E402
    EvalResult,
    plot_eval_curve,
    plot_eval_curve_from_dicts,
    plot_multi_horizon_grid,
    plot_bar_comparison,
    best_result,
    print_results_table,
)

# Time horizons for individual plots (separate files)
INDIVIDUAL_STEPS = [150, 200, 250, 300]

# Time horizons for the grid — matches DEFAULT_OVERLAP_CHECKPOINTS in soda/eval/metrics.py
# (125, 150, 175, 200, 225, 250, 275, 300); t=100 is not recorded by default
GRID_STEPS = list(range(125, 325, 25))  # [125, 150, 175, 200, 225, 250, 275, 300]


def _available_steps(raw_kernels: dict) -> set[int]:
    """Return all step values present as mean_score@{t} keys across any result."""
    steps = set()
    for results in raw_kernels.values():
        for r in results:
            for key in r:
                if key.startswith("mean_score@"):
                    steps.add(int(key.split("@")[1]))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to kernel_sweep_results.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent

    with open(data_path) as f:
        d = json.load(f)

    raw_kernels: dict[str, list[dict]] = d["kernels"]
    available = _available_steps(raw_kernels)

    # Build EvalResult runs for t=300 plots (uses mean_score = max over full episode)
    runs_t300: dict[str, list[EvalResult]] = {}
    for k_label, results in raw_kernels.items():
        runs_t300[k_label] = [
            EvalResult(
                epoch=r["epoch"],
                checkpoint=r["checkpoint"],
                n_action_steps=r["n_action_steps"],
                duration_termination=r["duration_termination"],
                n_episodes=r["n_episodes"],
                mean_score=r["mean_score"],
                std_score=r["std_score"],
                per_episode_scores=r["per_episode_scores"],
                run_label=k_label,
            )
            for r in results
        ]

    # --- Individual epoch-vs-score plots at each time horizon ---

    # t=300: use mean_score (max over full episode, most reliable)
    plot_eval_curve(
        runs_t300,
        output_dir / "plot_kernel_vs_epoch_t300.png",
        ylabel="Mean Max Overlap (%, t≤300)",
    )

    # t=250, t=200, t=150: use mean_score@{t} from raw dicts
    for t in [250, 200, 150]:
        if t not in available:
            print(f"  Skipping t={t}: no data available yet")
            continue
        plot_eval_curve_from_dicts(
            raw_kernels,
            output_dir / f"plot_kernel_vs_epoch_t{t}.png",
            score_key=f"mean_score@{t}",
            std_key=f"std_score@{t}",
            ylabel=f"Mean Max Overlap (%, t≤{t})",
        )

    # --- Subplot grid: all horizons from t=100 to t=300 ---
    grid_steps = [t for t in GRID_STEPS if t in available]
    if grid_steps:
        plot_multi_horizon_grid(
            raw_kernels,
            output_dir / "plot_kernel_vs_epoch_grid.png",
            steps=grid_steps,
            ncols=4,  # 8 steps → 2×4 grid
        )
    else:
        print("  Skipping grid: no step-level data available yet (run eval first)")

    # --- Bar chart: best epoch per kernel at t=300 ---
    best_per_kernel = {k: best_result(v) for k, v in runs_t300.items() if v}
    plot_bar_comparison(
        best_per_kernel,
        output_dir / "plot_kernel_best.png",
        ylabel="Mean Max Overlap (%, t≤300)",
    )

    print_results_table(runs_t300)
    print("Choose kernel with highest best-epoch mean_score for termination_study.")
    print("Update kernel_size in termination study configs (exp_term_*.yaml) before running.")


if __name__ == "__main__":
    main()
