"""Plot the LOVE training curve from the real training log.

Reads `love_training_log.txt` (raw stdout from
`modal run modal/modal_train_love.py`) and produces two plots:

    plot_love_val_loss.png      — val_loss vs iter (every 1000 iters)
    plot_love_train_breakdown.png — train loss / obs_cost / encoding_length
                                   per 100-iter logging tick

Both are real data from the Modal T4 training run that completed in ~3h
20m on 2026-06-07. The trained checkpoint itself was lost due to an
ephemeral-filesystem bug discovered shortly after — see
soda/option_discovery/unsupervised/love_adapter/README.md for the fix and
project_plan.md for the E3 status.

Run:
    python experiments/final_experiments/pusht/unsupervised_study/plot_love_training.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DEFAULT_LOG = HERE / "love_training_log.txt"


VAL_RE = re.compile(r"^iter\s+(\d+)\s+val_loss\s+([0-9.]+)")
TRAIN_RE = re.compile(
    r"^iter\s+(\d+)\s+loss\s+([0-9.]+)\s+obs_cost\s+([0-9.]+)\s+enc_len\s+([0-9.]+)"
)
BEST_RE = re.compile(r"→ new best, saved")


def parse_log(path: Path) -> dict:
    val_iters, val_losses = [], []
    new_best_iters = []
    train_iters, train_loss, obs_cost, enc_len = [], [], [], []

    last_val_iter = None
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            m = TRAIN_RE.match(line.lstrip())
            if m:
                train_iters.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                obs_cost.append(float(m.group(3)))
                enc_len.append(float(m.group(4)))
                continue
            m = VAL_RE.match(line.lstrip())
            if m:
                last_val_iter = int(m.group(1))
                val_iters.append(last_val_iter)
                val_losses.append(float(m.group(2)))
                continue
            if BEST_RE.search(line) and last_val_iter is not None:
                new_best_iters.append(last_val_iter)
                last_val_iter = None  # one best per val tick

    return {
        "val_iters": val_iters,
        "val_losses": val_losses,
        "new_best_iters": new_best_iters,
        "train_iters": train_iters,
        "train_loss": train_loss,
        "obs_cost": obs_cost,
        "enc_len": enc_len,
    }


def plot_val_loss(data: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=140)
    ax.plot(
        data["val_iters"],
        data["val_losses"],
        marker="o",
        markersize=5,
        linewidth=1.6,
        color="#1f77b4",
        label="LOVE validation objective",
    )
    val_map = dict(zip(data["val_iters"], data["val_losses"]))
    best_xs = [it for it in data["new_best_iters"] if it in val_map]
    best_ys = [val_map[it] for it in best_xs]
    if best_xs:
        ax.scatter(
            best_xs,
            best_ys,
            s=85,
            facecolor="none",
            edgecolor="#d62728",
            linewidth=1.6,
            zorder=5,
            label="checkpoint saved (new best)",
        )

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Validation objective\n(rec + KL + coding_len)")
    ax.set_title("LOVE training on Push-T (state-only, 25 650 demo frames)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=False, loc="upper right")

    best_val = min(data["val_losses"])
    best_iter = data["val_iters"][data["val_losses"].index(best_val)]
    ax.annotate(
        f"best: {best_val:.4f} @ iter {best_iter}",
        xy=(best_iter, best_val),
        xytext=(0.62, 0.55),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#444", lw=0.8),
    )

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"wrote {output}")


def plot_train_breakdown(data: dict, output: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.0), dpi=140, sharex=True)

    ax1.plot(
        data["train_iters"],
        data["train_loss"],
        color="#1f77b4",
        linewidth=0.9,
        alpha=0.55,
        label="total train loss",
    )
    ax1.plot(
        data["train_iters"],
        data["obs_cost"],
        color="#2ca02c",
        linewidth=1.0,
        label="action recon cross-entropy",
    )
    ax1.set_ylabel("loss")
    ax1.set_title("Training breakdown (logged every 100 iters)")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(frameon=False, loc="upper right")

    ax2.plot(
        data["train_iters"],
        data["enc_len"],
        color="#9467bd",
        linewidth=1.0,
        label="encoding_length (description-length term)",
    )
    ax2.set_xlabel("Training iteration")
    ax2.set_ylabel("encoding_length")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", type=Path, default=DEFAULT_LOG)
    p.add_argument("--out-dir", type=Path, default=HERE)
    args = p.parse_args()

    data = parse_log(args.log)
    print(
        f"parsed {len(data['val_iters'])} val ticks, "
        f"{len(data['train_iters'])} train ticks, "
        f"{len(data['new_best_iters'])} new-best events"
    )
    plot_val_loss(data, args.out_dir / "plot_love_val_loss.png")
    plot_train_breakdown(data, args.out_dir / "plot_love_train_breakdown.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
