"""
Speed comparison: Mean Max Overlap (%, t<=T) vs T for DP k=5, SODA k=5, SODA k=9.

Shows how quickly each policy achieves its peak overlap as the episode time budget grows.

Data sources:
  DP k=5  : comparison_study/dp_k5/eval_log.json  (mean_score@{t}, 50ep)
  SODA k=5: kernel_size_study/kernel_sweep_results.json, k5 best epoch (350)
  SODA k=9: kernel_size_study/kernel_sweep_results.json, k9 best epoch (450)

Usage:
  python experiments/final_experiments/pusht/comparison_study/plot_speed_comparison.py
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE           = Path(__file__).parent
KERNEL_RESULTS = HERE.parent / "kernel_size_study" / "kernel_sweep_results.json"
DP_EVAL_LOG    = HERE / "dp_k5" / "eval_log.json"
OUTPUT_PLOT    = HERE / "plot_speed_comparison.png"

TIMESTEPS = [125, 150, 175, 200, 225, 250, 275, 300]

# Visual style — consistent with comparison study plots
DP_COLOR     = "#7f7f7f"   # gray
SODA_K5_COLOR = "#ff7f0e"  # orange
SODA_K9_COLOR = "#1f77b4"  # blue


def _load_dp(path: Path) -> tuple[list[float], list[float]]:
    """Returns (means_pct, stds_pct) at each TIMESTEP from eval_log.json."""
    d = json.loads(path.read_text())
    m = d["metrics"]
    means = [m[f"mean_score@{t}"] * 100.0 for t in TIMESTEPS]
    stds  = [m[f"std_score@{t}"]  * 100.0 for t in TIMESTEPS]
    return means, stds


def _load_soda(kernel: str, epoch: int) -> tuple[list[float], list[float]]:
    """Returns (means_pct, stds_pct) at each TIMESTEP from kernel_sweep_results.json."""
    d = json.loads(KERNEL_RESULTS.read_text())
    runs = d["kernels"][kernel]
    r = next((r for r in runs if r["epoch"] == epoch), None)
    if r is None:
        available = [x["epoch"] for x in runs]
        raise ValueError(f"Epoch {epoch} not found for kernel={kernel}. Available: {available}")
    means = [r[f"mean_score@{t}"] for t in TIMESTEPS]
    stds  = [r.get(f"std_score@{t}", 0.0) for t in TIMESTEPS]
    return means, stds


def _soda_best_epoch(kernel: str) -> int:
    d = json.loads(KERNEL_RESULTS.read_text())
    runs = d["kernels"][kernel]
    return max(runs, key=lambda r: r["mean_score"])["epoch"]


def main() -> None:
    if not DP_EVAL_LOG.exists():
        print(f"ERROR: {DP_EVAL_LOG} not found.")
        print("Run: modal run modal/modal_eval.py --config configs/pusht/dp_frozen.yaml "
              "--n-test 50 --no-video "
              "--output-dir final_experiments/pusht/comparison_study/dp_k5 "
              "--local-save-path experiments/final_experiments/pusht/comparison_study/dp_k5")
        return

    k5_epoch = _soda_best_epoch("k5")
    k9_epoch = _soda_best_epoch("k9")

    dp_means,   dp_stds   = _load_dp(DP_EVAL_LOG)
    k5_means,   k5_stds   = _load_soda("k5", k5_epoch)
    k9_means,   k9_stds   = _load_soda("k9", k9_epoch)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(TIMESTEPS, dp_means, marker="o", color=DP_COLOR,      linewidth=1.8, markersize=6,
            label=f"Diffusion Policy (k=5, epch500)")
    ax.plot(TIMESTEPS, k5_means, marker="s", color=SODA_K5_COLOR, linewidth=1.8, markersize=6,
            label=f"SODA (k=5, epch{k5_epoch})")
    ax.plot(TIMESTEPS, k9_means, marker="^", color=SODA_K9_COLOR, linewidth=1.8, markersize=6,
            label=f"SODA (k=9, epch{k9_epoch})")

    ax.set_xlabel("Episode Time Budget T (steps)")
    ax.set_ylabel("Mean Max Overlap (%, t<=T, 50ep)")
    ax.set_xticks(TIMESTEPS)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 100)
    ax.set_xlim(TIMESTEPS[0] - 5, TIMESTEPS[-1] + 5)

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {OUTPUT_PLOT}")
    plt.close(fig)

    print("\n=== Speed comparison (mean max overlap %) ===")
    print(f"{'T':>5}  {'DP k=5':>9}  {'SODA k=5':>9}  {'SODA k=9':>9}")
    print("-" * 40)
    for i, t in enumerate(TIMESTEPS):
        print(f"{t:>5}  {dp_means[i]:>8.2f}%  {k5_means[i]:>8.2f}%  {k9_means[i]:>8.2f}%")


if __name__ == "__main__":
    main()
