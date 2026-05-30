"""
Plot stage3: joint runs vs best frozen from stage2.

Usage:
  python experiments/final_experiments/pusht/termination_study/stage3_joint/plot_stage3.py \\
    --stage3-data .../stage3_results.json \\
    --stage2-data .../stage2_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "shared"))
from experiment_utils import EvalResult, plot_bar_comparison, best_result  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-data", required=True)
    parser.add_argument("--stage2-data", required=True)
    args = parser.parse_args()

    with open(args.stage3_data) as f:
        s3 = json.load(f)
    with open(args.stage2_data) as f:
        s2 = json.load(f)

    # Build best-epoch results for bar chart
    results: dict[str, EvalResult] = {}

    def _best_from(run_results: list[dict], label: str) -> EvalResult:
        best = max(run_results, key=lambda r: r["mean_score"])
        return EvalResult(
            epoch=best["epoch"], checkpoint=best["checkpoint"],
            n_action_steps=best["n_action_steps"], duration_termination=best["duration_termination"],
            n_episodes=best["n_episodes"], mean_score=best["mean_score"],
            std_score=best["std_score"], per_episode_scores=best["per_episode_scores"],
            run_label=label,
        )

    # Stage 3 (joint)
    for run_label, run_results in s3["results"].items():
        if run_results:
            results[f"{run_label}\n(joint)"] = _best_from(run_results, run_label)

    # Stage 2 best frozen (for comparison)
    for run_label, run_results in s2["results"].items():
        if run_results:
            results[f"{run_label}\n(frozen)"] = _best_from(run_results, run_label)

    output_dir = Path(args.stage3_data).parent
    plot_bar_comparison(
        results,
        output_dir / "plot_stage3_bar.png",
        title="Stage 3: Joint vs Frozen Training",
    )

    print("\n=== Stage 3 summary ===")
    for label, r in sorted(results.items(), key=lambda x: x[1].mean_score, reverse=True):
        print(f"  {label}: mean={r.mean_score:.4f} ± {r.std_score:.4f}")


if __name__ == "__main__":
    main()
