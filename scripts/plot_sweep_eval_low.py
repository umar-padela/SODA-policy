"""
Plot sweep-eval results for a π_low checkpoint sweep.

Two subplots:
  Top:    train_loss_diffusion and val_loss_diffusion vs epoch (from metrics.json).
          Omitted when --train-log is not provided.
  Bottom: eval mean max overlap ± std per epoch (from sweep_summary.json).

Usage::

  # Eval scores only
  python scripts/plot_sweep_eval_low.py tmp/sweep_summary.json

  # With training loss overlay
  python scripts/plot_sweep_eval_low.py tmp/sweep_summary.json \\
      --train-log tmp/metrics.json

  # Custom output path
  python scripts/plot_sweep_eval_low.py tmp/sweep_summary.json \\
      --train-log tmp/metrics.json --output tmp/sweep_eval_low.png

To download metrics.json from the Modal volume::

  modal volume get soda-experiments \\
    pusht/train_low/soda_supervised/metrics.json tmp/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_sweep(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_train_log(path: str) -> list[dict]:
    """Read metrics.json (JSON array written by write_metrics_history)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot(
    sweep_path: str,
    train_log_path: str | None,
    output: str,
    title: str | None,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    sweep = _load_sweep(sweep_path)
    rows = sorted(sweep["rows"], key=lambda r: r["epoch"])

    epochs_eval = [r["epoch"] for r in rows]
    means = [r["mean_max_overlap"] for r in rows]
    stds = [r["std"] for r in rows]

    has_log = train_log_path is not None
    fig, axes = plt.subplots(
        2 if has_log else 1,
        1,
        figsize=(12, 9 if has_log else 5),
        sharex=False,
    )
    if not has_log:
        axes = [axes]  # uniform indexing

    run_label = Path(sweep["low_dir"]).name if "low_dir" in sweep else "π_low"
    suptitle = title or f"Sweep eval — {run_label}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    # ── Top: training / val loss ──────────────────────────────────────────────
    if has_log:
        ax_loss = axes[0]
        history = _load_train_log(train_log_path)
        ep_tr = [r["epoch"] for r in history]
        loss_tr = [r.get("train_loss_diffusion", r.get("train_loss")) for r in history]
        loss_val = [r.get("val_loss_diffusion", r.get("val_loss")) for r in history]

        ax_loss.plot(ep_tr, loss_tr, linewidth=1.0, color="#aec7e8", alpha=0.7,
                     label="train loss (diffusion)")
        ax_loss.plot(ep_tr, loss_val, linewidth=1.8, color="#1f77b4",
                     label="val loss (diffusion)")

        non_nan_val = [(e, v) for e, v in zip(ep_tr, loss_val) if v is not None and not (isinstance(v, float) and v != v)]
        if non_nan_val:
            best_ep, best_v = min(non_nan_val, key=lambda x: x[1])
            ax_loss.axvline(best_ep, color="#1f77b4", linestyle="--", alpha=0.5,
                            label=f"best val {best_v:.4f} @ ep {best_ep}")
            ax_loss.scatter([best_ep], [best_v], color="#1f77b4", zorder=5, s=50)

        # mark phase boundary if two-phase (p1 epochs from sweep metadata)
        p1_epochs = None
        if "rows" in sweep:
            # heuristic: if first pinned epoch exists and < min_epoch, use it as boundary
            pinned = sweep.get("pinned_epochs", [])
            p1_candidates = [e for e in pinned if e <= 150]
            if p1_candidates:
                p1_epochs = max(p1_candidates)
        if p1_epochs:
            ax_loss.axvline(p1_epochs, color="gray", linestyle=":", alpha=0.6,
                            label=f"phase boundary ep {p1_epochs}")

        ax_loss.set_ylabel("Diffusion loss (MSE)")
        ax_loss.set_yscale("log")
        ax_loss.legend(fontsize=8)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_title("Training & validation loss")

    # ── Bottom: eval score per epoch with ± std ───────────────────────────────
    ax_score = axes[-1]
    means_arr = np.array(means)
    stds_arr = np.array(stds)

    ax_score.errorbar(
        epochs_eval,
        means_arr,
        yerr=stds_arr,
        fmt="o-",
        linewidth=1.8,
        markersize=5,
        color="#2ca02c",
        ecolor="#98df8a",
        elinewidth=1.2,
        capsize=4,
        label="mean max overlap ± std",
    )

    best_idx = int(np.argmax(means_arr))
    ax_score.scatter(
        [epochs_eval[best_idx]], [means_arr[best_idx]],
        color="#d62728", zorder=6, s=80,
        label=f"best ep {epochs_eval[best_idx]}: {means_arr[best_idx]:.1f}%",
    )
    ax_score.axhline(means_arr[best_idx], color="#d62728", linestyle="--", alpha=0.4)

    ax_score.set_ylabel("Mean max overlap (%)")
    ax_score.set_xlabel("Epoch")
    ax_score.set_ylim(bottom=0)
    ax_score.legend(fontsize=8)
    ax_score.grid(True, alpha=0.3)
    ax_score.set_title(
        f"Sweep eval  |  high: {Path(sweep.get('high_checkpoint', '')).name}"
        f"  |  {sweep.get('n_episodes', '?')} ep/ckpt"
    )

    plt.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved: {out.resolve()}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_json", help="Path to sweep_summary.json from modal_sweep_eval_low.py")
    parser.add_argument("--train-log", default=None,
                        help="Path to metrics.json from train_low (adds loss panel)")
    parser.add_argument("--output", default="tmp/sweep_eval_low.png",
                        help="Output PNG path (default: tmp/sweep_eval_low.png)")
    parser.add_argument("--title", default=None, help="Custom figure title")
    args = parser.parse_args()

    plot(args.sweep_json, args.train_log, args.output, args.title)


if __name__ == "__main__":
    main()
