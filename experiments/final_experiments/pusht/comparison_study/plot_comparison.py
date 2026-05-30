"""
Plot final SODA vs DP baseline comparison.

Usage:
  python experiments/final_experiments/pusht/comparison_study/plot_comparison.py \\
    --data experiments/final_experiments/pusht/comparison_study/final_comparison.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "shared"))
from experiment_utils import EvalResult, plot_bar_comparison  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    with open(args.data) as f:
        d = json.load(f)

    results = {
        "SODA (best policy)": EvalResult(
            epoch=0, checkpoint=d["soda"]["checkpoint"],
            n_action_steps=d["soda"]["n_action_steps"],
            duration_termination=False, n_episodes=d["soda"]["n_episodes"],
            mean_score=d["soda"]["mean_score"], std_score=d["soda"]["std_score"],
            per_episode_scores=d["soda"]["per_episode_scores"],
        ),
        "DP Baseline": EvalResult(
            epoch=0, checkpoint=d["dp_baseline"]["checkpoint"],
            n_action_steps=d["soda"]["n_action_steps"],
            duration_termination=False, n_episodes=d["dp_baseline"]["n_episodes"],
            mean_score=d["dp_baseline"]["mean_score"], std_score=0.0,
            per_episode_scores=[],
        ),
    }

    plot_bar_comparison(
        results,
        Path(args.data).parent / "plot_final_comparison.png",
        title="Push-T: SODA Hierarchical Policy vs Diffusion Policy Baseline",
        figsize=(6, 5),
    )

    soda_mean = d["soda"]["mean_score"]
    dp_mean = d["dp_baseline"]["mean_score"]
    print(f"\nSODA:        {soda_mean:.4f} ± {d['soda']['std_score']:.4f}")
    print(f"DP Baseline: {dp_mean:.4f}")
    print(f"Improvement: {(soda_mean - dp_mean):+.4f} ({(soda_mean - dp_mean) / dp_mean * 100:+.1f}%)")


if __name__ == "__main__":
    main()
