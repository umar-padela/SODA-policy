"""
Stream-parse Columbia DP logs.json.txt files and produce two figures:

  Figure 1: 6 subplots — loss + success rate for each of train_0/1/2
  Figure 2: 2 subplots — all three seeds overlaid (loss, success rate)

Usage:
    python scripts/plot_dp_train_log.py
    python scripts/plot_dp_train_log.py --url-base <other base URL>
    python scripts/plot_dp_train_log.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://diffusion-policy.cs.columbia.edu/data/experiments/"
    "image/pusht/diffusion_policy_cnn"
)

SEED_COLORS = {
    0: ("#1f77b4", "#aec7e8"),   # blue shades
    1: ("#d62728", "#f7b6ba"),   # red shades
    2: ("#2ca02c", "#98df8a"),   # green shades
}


def stream_parse(source: str) -> tuple[list, list]:
    """Parse JSONL from a URL or local file → (train_rows, val_rows)."""
    train_rows, val_rows = [], []

    if source.startswith("http"):
        resp = urllib.request.urlopen(source)
        lines = (line.decode("utf-8") for line in resp)
    else:
        lines = Path(source).open(encoding="utf-8")

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue

        epoch = d.get("epoch", 0)
        step = d.get("global_step", 0)

        if "val_loss" in d:
            val_rows.append({
                "epoch": epoch,
                "step": step,
                "val_loss": d["val_loss"],
                "train_loss": d.get("train_loss"),
                "test_mean_score": d.get("test/mean_score"),
                "train_mean_score": d.get("train/mean_score"),
            })
        elif "train_loss" in d:
            train_rows.append({"epoch": epoch, "step": step, "train_loss": d["train_loss"]})

    return train_rows, val_rows


def load_all(seeds: list[int], url_base: str) -> dict[int, tuple[list, list]]:
    data = {}
    for s in seeds:
        url = f"{url_base}/train_{s}/logs.json.txt"
        print(f"Fetching train_{s}: {url}")
        tr, vr = stream_parse(url)
        print(f"  train steps: {len(tr)}  val checkpoints: {len(vr)}")
        data[s] = (tr, vr)
    return data


def _add_loss_panel(ax, train_rows, val_rows, title, color_dark, color_light):
    epochs_tr = [r["epoch"] for r in train_rows]
    loss_tr = [r["train_loss"] for r in train_rows]

    epochs_val = [r["epoch"] for r in val_rows]
    loss_val = [r["val_loss"] for r in val_rows]
    loss_tr_val = [r["train_loss"] for r in val_rows if r["train_loss"] is not None]
    epochs_tv = [r["epoch"] for r in val_rows if r["train_loss"] is not None]

    ax.plot(epochs_tr, loss_tr, alpha=0.25, linewidth=0.5, color=color_light)
    ax.plot(epochs_tv, loss_tr_val, linewidth=1.2, color=color_dark, alpha=0.6,
            label="train loss (at val)")
    ax.plot(epochs_val, loss_val, linewidth=1.8, color="tomato", label="val loss")

    if loss_val:
        best_idx = loss_val.index(min(loss_val))
        ax.axvline(epochs_val[best_idx], color="tomato", linestyle="--", alpha=0.6,
                   label=f"best val {min(loss_val):.4f} @ ep {epochs_val[best_idx]}")
        ax.scatter([epochs_val[best_idx]], [loss_val[best_idx]], color="tomato", zorder=5, s=50)

    ax.set_ylabel("Loss (diffusion MSE)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")


def _add_score_panel(ax, val_rows, color_dark, color_light):
    test_scores = [(r["epoch"], r["test_mean_score"]) for r in val_rows
                   if r.get("test_mean_score") is not None]
    train_scores = [(r["epoch"], r["train_mean_score"]) for r in val_rows
                    if r.get("train_mean_score") is not None]

    if test_scores:
        ep, sc = zip(*test_scores)
        ax.plot(ep, sc, linewidth=1.8, color=color_dark, label="test mean score")
    if train_scores:
        ep, sc = zip(*train_scores)
        ax.plot(ep, sc, linewidth=1.0, color=color_light, alpha=0.8, label="train mean score")

    ax.set_ylabel("Mean max overlap score")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)


def plot_per_seed(all_data: dict[int, tuple[list, list]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    seeds = sorted(all_data)
    fig, axes = plt.subplots(len(seeds), 2, figsize=(16, 5 * len(seeds)))
    fig.suptitle("DP training logs — per seed", fontsize=14, fontweight="bold")

    for row, s in enumerate(seeds):
        train_rows, val_rows = all_data[s]
        color_dark, color_light = SEED_COLORS[s]

        _add_loss_panel(axes[row, 0], train_rows, val_rows,
                        title=f"train_{s} — Loss", color_dark=color_dark, color_light=color_light)
        _add_score_panel(axes[row, 1], val_rows, color_dark=color_dark, color_light=color_light)
        axes[row, 1].set_title(f"train_{s} — Success Rate")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path.resolve()}")
    plt.show()


def plot_overlay(all_data: dict[int, tuple[list, list]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    seeds = sorted(all_data)
    fig, (ax_loss, ax_score) = plt.subplots(2, 1, figsize=(13, 10), sharex=False)
    fig.suptitle("DP training logs — all seeds overlay", fontsize=14, fontweight="bold")

    for s in seeds:
        train_rows, val_rows = all_data[s]
        color_dark, color_light = SEED_COLORS[s]
        label = f"train_{s}"

        epochs_val = [r["epoch"] for r in val_rows]
        loss_val = [r["val_loss"] for r in val_rows]
        loss_tr_val = [r["train_loss"] for r in val_rows if r["train_loss"] is not None]
        epochs_tv = [r["epoch"] for r in val_rows if r["train_loss"] is not None]

        ax_loss.plot(epochs_tv, loss_tr_val, linewidth=1.0, color=color_dark, alpha=0.5,
                     linestyle="--")
        ax_loss.plot(epochs_val, loss_val, linewidth=1.8, color=color_dark, label=f"{label} val loss")

        test_scores = [(r["epoch"], r["test_mean_score"]) for r in val_rows
                       if r.get("test_mean_score") is not None]
        if test_scores:
            ep, sc = zip(*test_scores)
            ax_score.plot(ep, sc, linewidth=1.8, color=color_dark, label=f"{label} test score")

    ax_loss.set_ylabel("Loss (diffusion MSE)")
    ax_loss.set_title("Val loss — all seeds")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("Epoch")

    ax_score.set_ylabel("Mean max overlap score")
    ax_score.set_title("Test success rate — all seeds")
    ax_score.legend(fontsize=9)
    ax_score.grid(True, alpha=0.3)
    ax_score.set_ylim(0, 1.05)
    ax_score.set_xlabel("Epoch")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path.resolve()}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url-base", default=BASE_URL)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    all_data = load_all(args.seeds, args.url_base)

    plot_per_seed(all_data, Path("tmp/dp_train_log_per_seed.png"))
    plot_overlay(all_data, Path("tmp/dp_train_log_overlay.png"))


if __name__ == "__main__":
    main()
