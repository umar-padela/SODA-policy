"""
Plot stage 1b results — termination_study stage 1b eval.

Outputs (same dir as --data):
  plot_stage1b_score_vs_epoch.png   — mean rollout score vs epoch per run
  plot_stage1b_best.png             — bar chart: best epoch per run

Usage:
  python experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/plot_stage1b.py \\
    --data experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/stage1b_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "shared"))
from experiment_utils import (  # noqa: E402
    EvalResult,
    plot_eval_curve,
    plot_bar_comparison,
    best_result,
    print_results_table,
    BOTTLENECK_COLOR,
    OBS_COLOR,
    term_run_style,
)

DISPLAY_NAMES = {
    "obs_positive":                      "Obs (completion)",
    "obs_positive_negative":             "Obs (completion+escape)",
    "bottleneck_ddim_positive_negative": "Bottleneck (DDIM-5, completion+escape)",
}

# k=9 epoch 450 with duration termination — shared backbone for all stage1b runs.
# Source: experiments/final_experiments/pusht/kernel_size_study/kernel_sweep_results.json
DURATION_BASELINE = {"Chunk Duration (k=9,epch450)": 97.97}

STYLES = {
    "Obs (completion)":                       term_run_style(OBS_COLOR,        "solid",  "s", True),
    "Obs (completion+escape)":                term_run_style(OBS_COLOR,        "dashed", "s", True),
    "Bottleneck (DDIM-5, completion+escape)": term_run_style(BOTTLENECK_COLOR, "dashed", "o", False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to stage1b_results.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent

    with open(data_path) as f:
        d = json.load(f)

    raw: dict[str, list[dict]] = d.get("results", {})

    if not any(raw.values()):
        print("No results yet — run modal_eval_stage1b.py first.")
        return

    runs: dict[str, list[EvalResult]] = {}
    for run_label, results in raw.items():
        display = DISPLAY_NAMES.get(run_label, run_label)
        runs[display] = [
            EvalResult(
                epoch=r["epoch"],
                checkpoint=r["checkpoint"],
                n_action_steps=r["n_action_steps"],
                duration_termination=r.get("duration_termination", False),
                n_episodes=r["n_episodes"],
                mean_score=r["mean_score"],
                std_score=r["std_score"],
                per_episode_scores=r["per_episode_scores"],
                run_label=display,
            )
            for r in results
        ]

    # Score vs epoch
    plot_eval_curve(
        runs,
        output_dir / "plot_stage1b_score_vs_epoch.png",
        ylabel="Mean Max Overlap (%, t≤300, 50ep)",
        hlines=DURATION_BASELINE,
        legend_title="Termination Method",
        legend_outside_bottom=True,
        styles=STYLES,
        ylim=(65, 100),
    )

    # Best epoch bar chart
    best_per_run = {label: best_result(results) for label, results in runs.items() if results}
    if best_per_run:
        plot_bar_comparison(
            best_per_run,
            output_dir / "plot_stage1b_best.png",
            ylabel="Mean Max Overlap (%, t≤300, 50ep)",
            hlines=DURATION_BASELINE,
            legend_title="Termination",
        )

    print_results_table(runs)

    winner_key = d.get("winner")
    if winner_key:
        winner_display = DISPLAY_NAMES.get(winner_key, winner_key)
        print(f"\nWINNER: {winner_display}")
        best = d.get("best_per_run", {}).get(winner_key, {})
        if best:
            print(f"  epoch={best.get('epoch')}, mean={best.get('mean_score', 0):.4f} ± {best.get('std_score', 0):.4f}")


if __name__ == "__main__":
    main()
