"""
Plot stage3 joint training results.

Outputs (same dir as --stage3-data):
  plot_stage3_score_vs_epoch.png  — score vs epoch, color-coded by input type
  plot_stage3_bar.png             — bar chart of best epoch per run (+ stage2 frozen if provided)

Visual encoding (same as stage1 combined plot):
  Bottleneck variants  — blue,   different linestyles
  Obs variants         — orange, different linestyles
  Both (concat)        — green,  solid
  Baseline             — gray dashed hline

Usage:
  python .../plot_stage3.py --stage3-data .../stage3_results.json
  python .../plot_stage3.py --stage3-data .../stage3_results.json \\
      --stage2-data .../stage2_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "shared"))
from experiment_utils import (  # noqa: E402
    EvalResult, plot_bar_comparison, best_result,
    BOTTLENECK_COLOR, OBS_COLOR, BOTH_COLOR, term_run_style,
)

BASELINE_COLOR = "gray"
BASELINE_SCORE = 97.97
BASELINE_LABEL = "Chunk Duration (k=9,epch450)"

# (run_key, display_label, style_kwargs)
# Color=feature, linestyle=signal, marker=feature, fill=source (expert=filled, DDIM-5=hollow)
RUN_SPECS = [
    ("bottleneck_expert_joint",                "Bottleneck (expert, completion, joint)",        term_run_style(BOTTLENECK_COLOR, "solid",  "o", True)),
    ("bottleneck_ddim_positive_joint",         "Bottleneck (DDIM-5, completion, joint)",        term_run_style(BOTTLENECK_COLOR, "solid",  "o", False)),
    ("bottleneck_ddim_positive_negative_joint","Bottleneck (DDIM-5, completion+escape, joint)", term_run_style(BOTTLENECK_COLOR, "dashed", "o", False)),
    ("obs_positive_joint",                     "Obs (completion, joint)",                       term_run_style(OBS_COLOR,        "solid",  "s", True)),
    ("obs_positive_negative_joint",            "Obs (completion+escape, joint)",                term_run_style(OBS_COLOR,        "dashed", "s", True)),
    ("both_joint",                             "Both (obs + bottleneck, joint)",                term_run_style(BOTH_COLOR,       "solid",  "D", True)),
]

DISPLAY_NAMES = {key: label for key, label, *_ in RUN_SPECS}


def _best_from(run_results: list[dict], label: str) -> EvalResult:
    best = max(run_results, key=lambda r: r["mean_score"])
    return EvalResult(
        epoch=best["epoch"], checkpoint=best["checkpoint"],
        n_action_steps=best["n_action_steps"], duration_termination=best["duration_termination"],
        n_episodes=best["n_episodes"], mean_score=best["mean_score"],
        std_score=best["std_score"], per_episode_scores=best["per_episode_scores"],
        run_label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-data", required=True)
    parser.add_argument("--stage2-data", default=None, help="Optional: adds frozen runs to bar chart")
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    with open(args.stage3_data) as f:
        s3 = json.load(f)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.stage3_data).parent
    results_by_key: dict[str, list[dict]] = s3.get("results", {})

    # --- Score vs epoch curve ---
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for run_key, label, style_kw in RUN_SPECS:
        results = results_by_key.get(run_key, [])
        if not results:
            continue
        results = sorted(results, key=lambda r: r["epoch"])
        epochs = [r["epoch"]      for r in results]
        means  = [r["mean_score"] for r in results]
        ax.plot(epochs, means, label=label, **style_kw)

    ax.axhline(BASELINE_SCORE, linestyle="--", linewidth=1.5,
               color=BASELINE_COLOR, label=BASELINE_LABEL)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Max Overlap (%, t≤300, 50ep)")
    ax.legend(title="Termination Method", loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=9, ncol=2, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(65, 100)

    out = output_dir / "plot_stage3_score_vs_epoch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out}")
    plt.close(fig)

    # --- Bar chart: best epoch per run (+ optional stage2 frozen comparison) ---
    bar_results: dict[str, EvalResult] = {}
    for run_key, label, *_ in RUN_SPECS:
        run_results = results_by_key.get(run_key, [])
        if run_results:
            bar_results[label] = _best_from(run_results, label)

    if args.stage2_data and Path(args.stage2_data).exists():
        with open(args.stage2_data) as f:
            s2 = json.load(f)
        for run_label, run_results in s2.get("results", {}).items():
            if run_results:
                disp = DISPLAY_NAMES.get(run_label.replace("_joint", ""), run_label)
                bar_results[f"{disp}\n(frozen)"] = _best_from(run_results, run_label)

    if bar_results:
        plot_bar_comparison(
            bar_results,
            output_dir / "plot_stage3_bar.png",
            ylabel="Mean Max Overlap (%, t≤300, 50ep)",
            hlines={BASELINE_LABEL: BASELINE_SCORE},
            legend_title="Termination",
        )

    print("\n=== Stage 3 summary ===")
    for run_key, label, *_ in RUN_SPECS:
        run_results = results_by_key.get(run_key, [])
        if run_results:
            best = max(run_results, key=lambda r: r["mean_score"])
            print(f"  {label}: best epoch={best['epoch']}, mean={best['mean_score']:.4f} ± {best['std_score']:.4f}")
        else:
            print(f"  {label}: no results yet")


if __name__ == "__main__":
    main()
