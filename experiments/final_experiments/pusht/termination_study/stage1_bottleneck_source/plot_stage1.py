"""
Plot stage 1a results — termination_study stage 1a eval.

Outputs (same dir as --data):
  plot_stage1_score_vs_epoch.png   — mean rollout score vs epoch per run
  plot_stage1_best.png             — bar chart: best epoch per run
  plot_stage1_combined.png         — all stage1a+1b runs on one plot (requires --stage1b-data)

Usage:
  python .../plot_stage1.py --data .../stage1_results.json
  python .../plot_stage1.py --data .../stage1_results.json \\
      --stage1b-data .../stage1b_results.json
"""

from __future__ import annotations
import argparse
import json
import subprocess
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
    term_run_style,
)

DISPLAY_NAMES = {
    "bottleneck_expert":        "Bottleneck (expert, completion)",
    "bottleneck_ddim_positive": "Bottleneck (DDIM-5, completion)",
}

# k=9 epoch 450 with duration termination — the backbone these runs fine-tune on top of.
# Source: experiments/final_experiments/pusht/kernel_size_study/kernel_sweep_results.json
DURATION_BASELINE = {"Chunk Duration (k=9,epch450)": 97.97}

STYLES = {
    "Bottleneck (expert, completion)":  term_run_style(BOTTLENECK_COLOR, "solid", "o", True),
    "Bottleneck (DDIM-5, completion)":  term_run_style(BOTTLENECK_COLOR, "solid", "o", False),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to stage1_results.json")
    parser.add_argument("--stage1b-data", default=None, help="Path to stage1b_results.json (enables combined plot)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent

    with open(data_path) as f:
        d = json.load(f)

    raw: dict[str, list[dict]] = d.get("results", {})

    if not any(raw.values()):
        print("No results yet — run modal_eval_stage1.py first.")
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

    plot_eval_curve(
        runs,
        output_dir / "plot_stage1_score_vs_epoch.png",
        ylabel="Mean Max Overlap (%, t≤300, 50ep)",
        hlines=DURATION_BASELINE,
        legend_title="Termination Method",
        legend_outside_bottom=True,
        styles=STYLES,
        ylim=(65, 100),
    )

    best_per_run = {label: best_result(results) for label, results in runs.items() if results}
    if best_per_run:
        plot_bar_comparison(
            best_per_run,
            output_dir / "plot_stage1_best.png",
            ylabel="Mean Max Overlap (%, t≤300, 50ep)",
            hlines=DURATION_BASELINE,
            legend_title="Termination",
        )

    print_results_table(runs)

    winner_key = d.get("winner")
    if winner_key:
        winner_display = DISPLAY_NAMES.get(winner_key, winner_key)
        print(f"\nWINNER: {winner_display}")
        print("Update BOTTLENECK_BEST_DIR/CONFIG in modal_eval_stage2.py.")

    if args.stage1b_data and Path(args.stage1b_data).exists():
        combined_script = Path(__file__).parent / "plot_stage1_combined.py"
        subprocess.run(
            [sys.executable, str(combined_script),
             "--stage1-data",  str(data_path),
             "--stage1b-data", args.stage1b_data,
             "--output-dir",   str(output_dir)],
            check=False,
        )


if __name__ == "__main__":
    main()
