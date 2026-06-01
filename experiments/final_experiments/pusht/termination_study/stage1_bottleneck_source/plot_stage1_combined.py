"""
Combined stage 1a + 1b score-vs-epoch plot.

All bottleneck and obs-based termination variants on one figure, plus the
chunk-length (duration termination) baseline as a horizontal reference line.

Visual encoding:
  Bottleneck variants  — blue,   different linestyles
  Obs variants         — orange, different linestyles
  Baseline             — gray dashed hline

No error bands (too many curves to read with shading).

Output:
  plot_stage1_combined.png  (saved alongside stage1_results.json)

Usage:
  python .../plot_stage1_combined.py \\
    --stage1-data  .../stage1_bottleneck_source/stage1_results.json \\
    --stage1b-data .../stage1b_obs_signal/stage1b_results.json
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "shared"))
from experiment_utils import BOTTLENECK_COLOR, OBS_COLOR, term_run_style  # noqa: E402

BASELINE_COLOR = "gray"
BASELINE_SCORE = 97.97
BASELINE_LABEL = "Chunk Duration (k=9,epch450)"

# (run_key, source, display_label, style_kwargs)
# Color=feature, linestyle=signal, marker=feature, fill=source (expert=filled, DDIM-5=hollow)
RUN_SPECS = [
    ("bottleneck_expert",                "stage1",  "Bottleneck (expert, completion)",        term_run_style(BOTTLENECK_COLOR, "solid",  "o", True)),
    ("bottleneck_ddim_positive",         "stage1",  "Bottleneck (DDIM-5, completion)",        term_run_style(BOTTLENECK_COLOR, "solid",  "o", False)),
    ("bottleneck_ddim_positive_negative","stage1b", "Bottleneck (DDIM-5, completion+escape)", term_run_style(BOTTLENECK_COLOR, "dashed", "o", False)),
    ("obs_positive",                     "stage1b", "Obs (completion)",                       term_run_style(OBS_COLOR,        "solid",  "s", True)),
    ("obs_positive_negative",            "stage1b", "Obs (completion+escape)",                term_run_style(OBS_COLOR,        "dashed", "s", True)),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-data",  required=True, help="Path to stage1_results.json")
    parser.add_argument("--stage1b-data", required=True, help="Path to stage1b_results.json")
    parser.add_argument("--output-dir",   default=None)
    args = parser.parse_args()

    stage1_path  = Path(args.stage1_data)
    stage1b_path = Path(args.stage1b_data)
    output_dir   = Path(args.output_dir) if args.output_dir else stage1_path.parent

    with open(stage1_path)  as f: s1  = json.load(f)
    with open(stage1b_path) as f: s1b = json.load(f)

    sources = {"stage1": s1.get("results", {}), "stage1b": s1b.get("results", {})}

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    for run_key, source, label, style_kw in RUN_SPECS:
        results = sources[source].get(run_key, [])
        if not results:
            continue
        results = sorted(results, key=lambda r: r["epoch"])
        epochs = [r["epoch"]      for r in results]
        means  = [r["mean_score"] for r in results]
        ax.plot(epochs, means, label=label, **style_kw)

    ax.axhline(BASELINE_SCORE, linestyle="--", linewidth=1.5, color=BASELINE_COLOR,
               label=BASELINE_LABEL)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Max Overlap (%, t≤300, 50ep)")
    ax.legend(title="Termination Method", loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=9, ncol=2, borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(65, 100)

    out = output_dir / "plot_stage1_combined.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
