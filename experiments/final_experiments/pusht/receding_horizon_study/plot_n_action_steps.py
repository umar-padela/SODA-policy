"""
Plot n_action_steps sweep results — receding_horizon_study.

Line plot: x = receding horizon window size, y = mean max overlap.
open_loop is shown as a horizontal dashed reference line (no fixed x position).

Output: plot_n_action_steps.png (same dir as --data)

Usage:
  python experiments/final_experiments/pusht/receding_horizon_study/plot_n_action_steps.py \\
    --data experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/sweep_results.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to sweep_results.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    with open(data_path) as f:
        sweep = json.load(f)

    import matplotlib.pyplot as plt

    # Separate numeric n_action_steps runs from open_loop
    numeric = {
        int(r["n_action_steps"]): r
        for key, r in sweep.items()
        if r.get("n_action_steps") is not None
    }
    open_loop = sweep.get("open_loop")

    xs     = sorted(numeric.keys())
    means  = [numeric[n]["mean_score"] for n in xs]
    stds   = [numeric[n]["std_score"]  for n in xs]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(xs, means, yerr=stds, fmt="o-", capsize=4,
                linewidth=1.8, markersize=6, label="Duration termination")

    if open_loop:
        ax.axhline(open_loop["mean_score"], linestyle="--", linewidth=1.5,
                   color="gray", label="Open loop")

    ax.set_xlabel("Receding Horizon (n_action_steps)")
    ax.set_ylabel("Mean Max Overlap (%, t≤300, 50ep)")
    ax.set_xticks(xs)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    output = Path(args.output) if args.output else data_path.parent / "plot_n_action_steps.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {output}")
    plt.close(fig)

    print("\n=== n_action_steps results ===")
    for n in xs:
        r = numeric[n]
        print(f"  n={n:>3}: {r['mean_score']:.4f} ± {r['std_score']:.4f}")
    if open_loop:
        print(f"  open_loop: {open_loop['mean_score']:.4f} ± {open_loop['std_score']:.4f}")

    best_n = max(xs, key=lambda n: numeric[n]["mean_score"])
    print(f"\nBest n_action_steps: {best_n} (mean={numeric[best_n]['mean_score']:.4f})")


if __name__ == "__main__":
    main()
