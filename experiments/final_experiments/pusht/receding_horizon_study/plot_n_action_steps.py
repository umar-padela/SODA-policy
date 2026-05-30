"""
Plot n_action_steps sweep results — receding_horizon_study step 3.

Usage:
  python experiments/final_experiments/pusht/receding_horizon_study/plot_n_action_steps.py \\
    --data experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/sweep_results.json \\
    --output experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/plot_n_action_steps.png
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "shared"))
from experiment_utils import plot_bar_comparison, EvalResult  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to sweep_results.json")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: same dir as data)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    with open(data_path) as f:
        sweep = json.load(f)

    # Build EvalResult dict for bar chart
    results: dict[str, EvalResult] = {}
    for label, r in sweep.items():
        results[label] = EvalResult(
            epoch=0,
            checkpoint=r.get("low_checkpoint", ""),
            n_action_steps=r.get("n_action_steps") or 0,
            duration_termination=True,
            n_episodes=r.get("n_episodes", 0),
            mean_score=r["mean_score"],
            std_score=r["std_score"],
            per_episode_scores=r.get("per_episode_scores", []),
            run_label=label,
            n_action_steps_label=label,
        )

    output_path = args.output or str(data_path.parent / "plot_n_action_steps.png")
    plot_bar_comparison(
        results,
        output_path,
        title="Push-T: n_action_steps Sweep (k=5, duration_termination)",
        ylabel="Mean Max Overlap Score",
    )

    # Print table
    print("\n=== n_action_steps results ===")
    for label, r in sorted(results.items(), key=lambda x: x[1].mean_score, reverse=True):
        print(f"  {label:>12}: {r.mean_score:.4f} ± {r.std_score:.4f}")

    best = max(results.values(), key=lambda r: r.mean_score)
    print(f"\nRecommended n_action_steps: {best.n_action_steps_label} (mean={best.mean_score:.4f})")


if __name__ == "__main__":
    main()
