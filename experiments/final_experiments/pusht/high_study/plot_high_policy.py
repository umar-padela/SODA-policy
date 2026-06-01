"""
Plot high policy study results.

Outputs:
  plot_high_policy_speed.png  — mean max overlap vs T (speed curve)
  plot_high_policy_bar.png    — bar chart: overall mean at t<=300

Usage:
  python experiments/final_experiments/pusht/high_study/plot_high_policy.py \
    --data experiments/final_experiments/pusht/high_study/high_policy_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "shared"))
from experiment_utils import BOTTLENECK_COLOR, OBS_COLOR  # noqa: E402

HERE      = Path(__file__).parent
TIMESTEPS = [125, 150, 175, 200, 225, 250, 275, 300]

BASELINE_COLOR    = "#7f7f7f"    # gray  — segment-start (existing)
ALL_FRAMES_COLOR  = "#2ca02c"    # green — all-frames (new)

STYLES = {
    "baseline":   {"color": BASELINE_COLOR,   "marker": "o", "linestyle": "solid",  "linewidth": 1.8, "markersize": 6},
    "all_frames": {"color": ALL_FRAMES_COLOR,  "marker": "s", "linestyle": "dashed", "linewidth": 1.8, "markersize": 6},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    data_path  = Path(args.data)
    output_dir = Path(args.output_dir) if args.output_dir else data_path.parent

    d       = json.loads(data_path.read_text())
    results = d.get("results", {})

    if not results:
        print("No results yet.")
        return

    # --- Speed curve: mean overlap vs T ---
    fig, ax = plt.subplots(figsize=(7, 5))

    for run_key, r in results.items():
        style = STYLES.get(run_key, {})
        horizons = r.get("horizon_means", {})
        means = [horizons.get(str(t), horizons.get(t, float("nan"))) for t in TIMESTEPS]
        ax.plot(TIMESTEPS, means, label=r["label"], **style)

    ax.set_xlabel("Episode Time Budget T (steps)")
    ax.set_ylabel("Mean Max Overlap (%, t<=T, 50ep)")
    ax.set_xticks(TIMESTEPS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(60, 100)
    ax.set_xlim(TIMESTEPS[0] - 5, TIMESTEPS[-1] + 5)

    out = output_dir / "plot_high_policy_speed.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out}")
    plt.close(fig)

    # --- Bar chart: overall mean at t<=300 ---
    fig, ax = plt.subplots(figsize=(5, 4))

    labels = [r["label"] for r in results.values()]
    means  = [r["mean_score"] for r in results.values()]
    stds   = [float(np.std(r["per_episode_scores"])) for r in results.values()]
    colors = [STYLES.get(k, {}).get("color", "steelblue") for k in results]

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Max Overlap (%, t<=300, 50ep)")
    ax.set_ylim(60, 100)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)

    out = output_dir / "plot_high_policy_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out}")
    plt.close(fig)

    # --- Summary ---
    print("\n=== High policy study summary ===")
    for run_key, r in results.items():
        print(f"  {r['label']}: mean={r['mean_score']:.2f}%  n={r['n_episodes']}")


if __name__ == "__main__":
    main()
