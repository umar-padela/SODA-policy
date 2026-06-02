"""
Noise robustness comparison — grouped bar chart: DP vs BID vs SODA vs SODA-BID.

Reads JSON files produced by modal_eval_bid_comparison.py.
Each JSON has schema: {policy, noise_eta, episodes: [{seed, score}], mean_score, std_score}.

Saves plot_noise_study.png alongside this script.

Usage:
  python experiments/final_experiments/pusht/noise_study/plot_noise_study.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent

POLICIES = {
    "dp":       {"label": "DP (k=5)",    "color": "#1f77b4"},
    "bid":      {"label": "BID",         "color": "#2ca02c"},
    "soda":     {"label": "SODA",        "color": "#ff7f0e"},
    "soda_bid": {"label": "SODA-BID",    "color": "#d62728"},
}
NOISE_LABELS = {0.0: "η = 0.0\n(no noise)", 1.0: "η = 1.0\n(moderate)"}


def _load(policy: str, eta: float) -> dict | None:
    path = HERE / f"{policy}_eta{eta}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    # Support both formats: episodes list or flat per_episode_scores
    if "episodes" in d and d["episodes"] and "score" in d["episodes"][0]:
        scores = [ep["score"] for ep in d["episodes"]]
        d["mean_score"] = float(np.mean(scores))
        d["std_score"] = float(np.std(scores))
        d["n_episodes"] = len(scores)
    return d


def main() -> None:
    etas = sorted(NOISE_LABELS.keys())

    data: dict[str, dict[float, dict]] = {p: {} for p in POLICIES}
    for policy in POLICIES:
        for eta in etas:
            d = _load(policy, eta)
            if d:
                data[policy][eta] = d

    if not any(data[p] for p in POLICIES):
        print("No result JSON files found. Run modal_eval_bid_comparison.py first.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    n_groups = len(etas)
    n_policies = len(POLICIES)
    width = 0.18
    x = np.arange(n_groups)

    for i, (policy, style) in enumerate(POLICIES.items()):
        offset = (i - (n_policies - 1) / 2) * (width + 0.02)
        means, stds, ns = [], [], []
        for eta in etas:
            d = data[policy].get(eta)
            if d:
                means.append(d["mean_score"] * 100.0)
                stds.append(d["std_score"] * 100.0)
                ns.append(d.get("n_episodes", 0))
            else:
                means.append(0.0)
                stds.append(0.0)
                ns.append(0)

        bars = ax.bar(
            x + offset,
            means,
            width,
            label=style["label"],
            color=style["color"],
            alpha=0.85,
            yerr=stds,
            capsize=3,
            error_kw={"elinewidth": 1.0, "ecolor": "black", "alpha": 0.6},
        )
        for j, (bar, mean, std, n) in enumerate(zip(bars, means, stds, ns)):
            if n == 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 2.0, "n/a",
                        ha="center", va="bottom", fontsize=7, color="gray")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + std + 1.5,
                        f"{mean:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([NOISE_LABELS[e] for e in etas], fontsize=10)
    ax.set_ylabel("Max Block-Target Overlap (%)", fontsize=11)
    ax.set_title("Noise Robustness: DP vs BID vs SODA vs SODA-BID (Push-T)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    out = HERE / "plot_noise_study.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")

    # Print table
    eta_headers = "  ".join(f"{'eta='+str(e):>10}" for e in etas)
    header = f"{'Policy':<12}  {eta_headers}"
    print(f"\n{header}")
    print("-" * (len(header) + 4))
    for policy, style in POLICIES.items():
        row = f"{style['label']:<12}  "
        vals = []
        for eta in etas:
            d = data[policy].get(eta)
            vals.append(f"{d['mean_score']*100:.1f}%" if d else "   n/a")
        row += "  ".join(f"{v:>10}" for v in vals)
        print(row)


if __name__ == "__main__":
    main()
