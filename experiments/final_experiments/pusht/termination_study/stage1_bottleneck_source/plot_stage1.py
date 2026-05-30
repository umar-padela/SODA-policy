"""
Plot stage1 results: bottleneck_expert vs bottleneck_ddim_positive.

Usage:
  python experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/plot_stage1.py \\
    --data experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/stage1_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "shared"))
from experiment_utils import EvalResult, plot_eval_curve, plot_bar_comparison, best_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    data_path = Path(args.data)
    with open(data_path) as f:
        d = json.load(f)

    runs: dict[str, list[EvalResult]] = {}
    for run_label, results in d["results"].items():
        runs[run_label] = [
            EvalResult(
                epoch=r["epoch"],
                checkpoint=r["checkpoint"],
                n_action_steps=r["n_action_steps"],
                duration_termination=r["duration_termination"],
                n_episodes=r["n_episodes"],
                mean_score=r["mean_score"],
                std_score=r["std_score"],
                per_episode_scores=r["per_episode_scores"],
                run_label=run_label,
            )
            for r in results
        ]

    output_dir = data_path.parent
    plot_eval_curve(
        runs,
        output_dir / "plot_stage1_curve.png",
        title="Stage 1: Bottleneck Source (expert vs DDIM-5)",
    )

    best_per_run = {k: best_result(v) for k, v in runs.items() if v}
    plot_bar_comparison(
        best_per_run,
        output_dir / "plot_stage1_bar.png",
        title="Stage 1: Best Epoch — Bottleneck Source",
    )

    print(f"\nWinner: {d.get('winner', 'unknown')}")
    print("Update stage2 exp_term_both.yaml and stage3 configs with the winner's bottleneck source.")


if __name__ == "__main__":
    main()
