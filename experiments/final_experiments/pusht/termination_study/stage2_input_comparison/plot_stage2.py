"""
Plot stage2: obs vs bottleneck_best vs both vs duration_termination baseline.

Usage:
  python experiments/final_experiments/pusht/termination_study/stage2_input_comparison/plot_stage2.py \\
    --data experiments/final_experiments/pusht/termination_study/stage2_input_comparison/stage2_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "shared"))
from experiment_utils import (  # noqa: E402
    EvalResult, plot_eval_curve, plot_bar_comparison, best_result,
    BOTTLENECK_COLOR, OBS_COLOR, BOTH_COLOR, term_run_style,
)

DISPLAY_NAMES = {
    "bottleneck_best": "Bottleneck (stage 1a best)",
    "obs_best":        "Obs (stage 1b best)",
    "both":            "Both (obs + bottleneck)",
}

# bottleneck_best = bottleneck_expert winner; obs_best = obs_positive winner
STYLES = {
    "Bottleneck (stage 1a best)": term_run_style(BOTTLENECK_COLOR, "solid", "o", True),
    "Obs (stage 1b best)":        term_run_style(OBS_COLOR,        "solid", "s", True),
    "Both (obs + bottleneck)":    term_run_style(BOTH_COLOR,       "solid", "D", True),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    data_path = Path(args.data)
    with open(data_path) as f:
        d = json.load(f)

    runs: dict[str, list[EvalResult]] = {}
    for run_label, results in d["results"].items():
        display = DISPLAY_NAMES.get(run_label, run_label)
        runs[display] = [
            EvalResult(
                epoch=r["epoch"], checkpoint=r["checkpoint"],
                n_action_steps=r["n_action_steps"], duration_termination=False,
                n_episodes=r["n_episodes"], mean_score=r["mean_score"],
                std_score=r["std_score"], per_episode_scores=r["per_episode_scores"],
                run_label=display,
            )
            for r in results
        ]

    output_dir = data_path.parent
    plot_eval_curve(
        runs,
        output_dir / "plot_stage2_curve.png",
        hlines={"Chunk Duration (k=9,epch450)": 97.97},
        legend_title="Termination Method",
        legend_outside_bottom=True,
        styles=STYLES,
        ylim=(65, 100),
    )

    # Bar chart comparing best epoch per run + baseline
    best_per_run: dict[str, EvalResult] = {k: best_result(v) for k, v in runs.items() if v}

    baseline = d.get("duration_termination_baseline")
    if baseline:
        best_per_run["Chunk Duration (k=9,epch450)"] = EvalResult(
            epoch=0, checkpoint="", n_action_steps=d["n_action_steps"],
            duration_termination=True, n_episodes=baseline["n_episodes"],
            mean_score=baseline["mean_score"], std_score=baseline["std_score"],
            per_episode_scores=[], run_label="duration_termination",
        )

    plot_bar_comparison(
        best_per_run,
        output_dir / "plot_stage2_bar.png",
    )

    print("\n=== Stage 2 summary ===")
    for label, r in best_per_run.items():
        print(f"  {label}: mean={r.mean_score:.4f} ± {r.std_score:.4f}")


if __name__ == "__main__":
    main()
