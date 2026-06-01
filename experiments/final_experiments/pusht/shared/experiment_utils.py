"""
Shared utilities for SODA final experiments: plotting, data loading, eval wrappers.

All experiments write summary.json with a common schema (see EvalResult).
Plot functions accept dicts of {run_label: [EvalResult]} for easy multi-run comparison.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """One evaluation run (one epoch checkpoint, one n_action_steps setting)."""
    epoch: int
    checkpoint: str
    n_action_steps: int
    duration_termination: bool
    n_episodes: int
    mean_score: float
    std_score: float
    per_episode_scores: list[float]

    # Optional metadata for labeling
    run_label: str = ""
    n_action_steps_label: str = ""   # e.g. "open_loop" for open-loop mode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_eval_result(path: str | Path) -> EvalResult:
    """Load a single summary.json -> EvalResult."""
    with open(path) as f:
        d = json.load(f)
    return EvalResult(
        epoch=d["epoch"],
        checkpoint=d["checkpoint"],
        n_action_steps=d["n_action_steps"],
        duration_termination=d["duration_termination"],
        n_episodes=d["n_episodes"],
        mean_score=d["mean_score"],
        std_score=d["std_score"],
        per_episode_scores=d["per_episode_scores"],
        run_label=d.get("run_label", ""),
        n_action_steps_label=d.get("n_action_steps_label", ""),
    )


def save_eval_result(result: EvalResult, path: str | Path) -> None:
    """Write EvalResult -> summary.json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_eval_results(directory: str | Path) -> list[EvalResult]:
    """Load all summary.json files in a directory tree, sorted by epoch."""
    results = []
    for p in sorted(Path(directory).rglob("summary.json")):
        try:
            results.append(load_eval_result(p))
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
    results.sort(key=lambda r: r.epoch)
    return results


def save_sweep_results(
    results: dict[str, list[EvalResult]],
    path: str | Path,
) -> None:
    """Save {run_label: [EvalResult]} -> JSON for easy replotting."""
    serializable = {
        label: [r.to_dict() for r in run_results]
        for label, run_results in results.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved sweep results -> {path}")


def load_sweep_results(path: str | Path) -> dict[str, list[EvalResult]]:
    """Load {run_label: [EvalResult]} from a sweep JSON."""
    with open(path) as f:
        d = json.load(f)
    return {
        label: [
            EvalResult(**r) for r in run_results
        ]
        for label, run_results in d.items()
    }


# ---------------------------------------------------------------------------
# Termination study visual encoding (shared across all termination plot scripts)
# ---------------------------------------------------------------------------

BOTTLENECK_COLOR = "#1f77b4"   # blue   — all bottleneck variants
OBS_COLOR        = "#ff7f0e"   # orange — all obs variants
BOTH_COLOR       = "#2ca02c"   # green  — both (obs + bottleneck combined)


def term_run_style(color: str, linestyle: str, marker: str, filled: bool) -> dict:
    """Canonical matplotlib plot kwargs for termination study visual encoding.

    Color     = feature type  (bottleneck=blue, obs=orange).
    Linestyle = signal type   (completion=solid, completion+escape=dashed).
    Marker    = feature type  (bottleneck=circle 'o', obs=square 's').
    Fill      = action source (expert=filled, DDIM-5=hollow).
    """
    kw: dict = {
        "color":     color,
        "linestyle": linestyle,
        "marker":    marker,
        "linewidth": 1.8,
        "markersize": 6,
    }
    if not filled:
        kw["markerfacecolor"] = "none"
        kw["markeredgecolor"] = color
        kw["markeredgewidth"] = 1.5
    return kw


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _draw_eval_curve(
    ax,
    runs: dict[str, list],
    *,
    score_key: str = "mean_score",
    std_key: str = "std_score",
    epoch_key: str = "epoch",
    use_evalresult: bool = True,
    show_bands: bool = True,
    origin: tuple[int, float] | None = None,
    styles: dict[str, dict] | None = None,
) -> None:
    """Internal: draw lines into an existing axes. Works with EvalResult or raw dicts.

    origin: if given as (epoch, score), prepends a shared starting point to every run.
    styles: optional dict mapping run label -> extra matplotlib plot kwargs (color,
            linestyle, marker, markerfacecolor, etc.).  Per-label keys override defaults.
    """
    for label, results in runs.items():
        if not results:
            continue
        if use_evalresult:
            filtered = results
            epochs = [r.epoch for r in filtered]
            means = [r.mean_score for r in filtered]
            stds = [r.std_score for r in filtered]
        else:
            filtered = [r for r in results if score_key in r]
            if not filtered:
                continue
            epochs = [r[epoch_key] for r in filtered]
            means = [r[score_key] for r in filtered]
            stds = [r.get(std_key, 0.0) for r in filtered]

        if origin is not None:
            epochs = [origin[0]] + epochs
            means  = [origin[1]] + means
            stds   = [0.0]       + stds

        plot_kw: dict = {"marker": "o", "linewidth": 1.8, "markersize": 6}
        plot_kw.update((styles or {}).get(label, {}))
        line = ax.plot(epochs, means, label=label, **plot_kw)[0]
        if show_bands:
            ax.fill_between(
                epochs,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2,
                color=line.get_color(),
            )


def plot_eval_curve(
    runs: dict[str, list[EvalResult]],
    output_path: str | Path,
    *,
    title: str | None = None,
    xlabel: str = "Epoch",
    ylabel: str = "Mean Max Overlap (%, t≤300, 50ep)",
    figsize: tuple[float, float] = (8, 5),
    show: bool = False,
    hlines: dict[str, float] | None = None,
    legend_title: str | None = None,
    legend_outside_bottom: bool = False,
    legend_ncols: int = 1,
    show_bands: bool = False,
    origin: tuple[int, float] | None = None,
    xlim_left: float | None = None,
    styles: dict[str, dict] | None = None,
    ylim: tuple[float, float] = (0, 100),
) -> None:
    """Line plot: mean_score ± std_score vs epoch for multiple runs.

    hlines: optional dict of {label: mean_score} drawn as dashed horizontal baselines.
    legend_title: optional title shown above the legend.
    legend_outside_bottom: place legend below the axes, left-aligned.
    legend_ncols: columns in the outside-bottom legend.
    show_bands: set True to show ±std shaded bands.
    origin: if given as (epoch, score), all runs share this starting point (fine-tuning plots).
    xlim_left: if set, pin the x-axis left edge (e.g. 0 so a baseline hline shows from epoch 0).
    styles: optional dict mapping run label -> matplotlib plot kwargs (see term_run_style).
    ylim: y-axis (bottom, top). Default (0, 100). Use e.g. (65, 100) for termination study.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    _draw_eval_curve(ax, runs, use_evalresult=True, show_bands=show_bands, origin=origin, styles=styles)
    if hlines:
        for label, val in hlines.items():
            ax.axhline(val, linestyle="--", linewidth=1.5, color="gray", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend_outside_bottom:
        ax.legend(title=legend_title, loc="upper center", bbox_to_anchor=(0.5, -0.15),
                  fontsize=9, ncol=legend_ncols, borderaxespad=0)
    else:
        ax.legend(title=legend_title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(*ylim)
    if origin is not None:
        ax.set_xlim(left=0)
    if xlim_left is not None:
        ax.set_xlim(left=xlim_left)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_eval_curve_from_dicts(
    runs: dict[str, list[dict]],
    output_path: str | Path | None,
    *,
    score_key: str = "mean_score",
    std_key: str = "std_score",
    xlabel: str = "Epoch",
    ylabel: str = "Mean Max Overlap (%)",
    figsize: tuple[float, float] = (8, 5),
    show: bool = False,
    ax=None,
    ylim: tuple[float, float] = (0, 100),
) -> None:
    """
    Line plot using raw result dicts, supporting arbitrary score keys
    (e.g. mean_score@150, mean_score@250).

    Pass ax= to draw into an existing subplot axes instead of saving a file.
    """
    import matplotlib.pyplot as plt

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    _draw_eval_curve(ax, runs, score_key=score_key, std_key=std_key, use_evalresult=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(*ylim)

    if own_fig and output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {output_path}")
        if show:
            plt.show()
        plt.close(fig)


def plot_multi_horizon_grid(
    runs: dict[str, list[dict]],
    output_path: str | Path,
    *,
    steps: list[int],
    ncols: int = 3,
    figsize_per_cell: tuple[float, float] = (5, 3.8),
    show: bool = False,
    ylim: tuple[float, float] = (0, 100),
) -> None:
    """
    Grid of subplots, one per time horizon. Each subplot shows mean max overlap
    vs epoch for all runs at that horizon cutoff.

    steps: list of time steps to plot, e.g. [100, 125, 150, 175, 200, 225, 250, 275, 300]
    """
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(steps) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )

    for idx, t in enumerate(steps):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        score_key = f"mean_score@{t}"
        std_key = f"std_score@{t}"

        _draw_eval_curve(ax, runs, score_key=score_key, std_key=std_key, use_evalresult=False)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(f"Max Overlap (%, t≤{t})", fontsize=9)
        ax.set_title(f"t ≤ {t}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=8)

        # Only show legend on first cell
        if idx == 0:
            ax.legend(fontsize=8)
        else:
            legend = ax.get_legend()
            if legend:
                legend.remove()

    # Hide unused axes
    for idx in range(len(steps), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_bar_comparison(
    results: dict[str, EvalResult],
    output_path: str | Path,
    *,
    title: str | None = None,
    ylabel: str = "Mean Max Overlap (%, t≤300, 50ep)",
    figsize: tuple[float, float] = (8, 5),
    show: bool = False,
    hlines: dict[str, float] | None = None,
    legend_title: str | None = None,
) -> None:
    """Bar chart: one bar per run label, height = mean_score, error bar = std_score.

    hlines: optional dict of {label: mean_score} drawn as dashed horizontal baselines.
    legend_title: optional title shown above the legend (only shown when hlines are present).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(results.keys())
    means = [results[k].mean_score for k in labels]
    stds = [results[k].std_score for k in labels]

    auto_width = max(3, len(labels) * 1.4)
    fig, ax = plt.subplots(figsize=(auto_width, figsize[1]))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{mean:.1f}%",
            ha="center", va="bottom", fontsize=9,
        )

    if hlines:
        for label, val in hlines.items():
            ax.axhline(val, linestyle="--", linewidth=1.5, color="gray", label=label)
        ax.legend(fontsize=8, title=legend_title)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def best_result(results: list[EvalResult]) -> EvalResult:
    """Return the EvalResult with the highest mean_score."""
    return max(results, key=lambda r: r.mean_score)


def print_results_table(runs: dict[str, list[EvalResult]]) -> None:
    """Print a summary table of best epoch per run."""
    print(f"\n{'Run':<35} {'Best Epoch':>10} {'Mean Score':>12} {'Std':>8}")
    print("-" * 70)
    for label, results in runs.items():
        if not results:
            print(f"{label:<35} {'N/A':>10} {'N/A':>12} {'N/A':>8}")
            continue
        best = best_result(results)
        print(f"{label:<35} {best.epoch:>10} {best.mean_score:>12.4f} {best.std_score:>8.4f}")
    print()
