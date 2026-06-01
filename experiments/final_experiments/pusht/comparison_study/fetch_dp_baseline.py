"""
Download Columbia Diffusion Policy (CNN) Push-T training metrics and plot score vs epoch.

Downloads logs.json.txt for each seed (train_0, train_1, train_2) from the Columbia DP
experiment server, extracts per-epoch test/mean_score, aggregates across seeds, and
produces:

  dp_baseline_results.json  — structured results (same schema as other eval JSONs;
                               loadable by load_sweep_results / plot_eval_curve)
  plot_dp_baseline.png       — mean ± cross-seed std vs epoch (same style as our plots)

Framing: this IS our baseline experiment for the comparison_study. The plot and JSON
produced here feed directly into plot_comparison.py alongside SODA results.

Downloads are cached in dp_baseline_logs/ to avoid re-downloading (each ~50 MB).

Usage:
  python experiments/final_experiments/pusht/comparison_study/fetch_dp_baseline.py
  python .../fetch_dp_baseline.py --no-plot   # skip matplotlib if not installed
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE_DIR = HERE / "dp_baseline_logs"

OUTPUT_JSON = HERE / "diffusion_policy_k5_results.json"
OUTPUT_PLOT = HERE / "plot_dp_baseline.png"

sys.path.insert(0, str(HERE.parent / "shared"))
from experiment_utils import EvalResult, save_sweep_results, plot_eval_curve  # noqa: E402

BASE_URL = (
    "https://diffusion-policy.cs.columbia.edu"
    "/data/experiments/image/pusht/diffusion_policy_cnn"
)
SEEDS = ["train_0"]   # one seed, matches SODA training (single run)
MAX_EPOCH = 500       # matches SODA kernel sweep epoch budget

# Columbia logs use slash-separated keys e.g. "test/mean_score"
SCORE_KEYS = ["test/mean_score", "test_mean_score"]
STD_KEYS   = ["test/std_score",  "test_std_score"]

RUN_LABEL = "Diffusion Policy (CNN)"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_log(seed: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"{seed}_logs.json.txt"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"  [cached] {dest.name}  ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    url = f"{BASE_URL}/{seed}/logs.json.txt"
    print(f"  Downloading {url} ...", flush=True)

    def _progress(count, block_size, total):
        pct = min(100, count * block_size * 100 // total) if total > 0 else 0
        print(f"\r    {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\r  Done  ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def _get(obj: dict, keys: list[str], default=None):
    for k in keys:
        if k in obj:
            return obj[k]
    return default


def _parse_log(path: Path) -> dict[int, dict]:
    """
    Stream-parse JSONL log. Returns {epoch: {mean_score_pct, std_score_pct, per_episode_scores}}.
    Scores are converted from [0, 1] → %.
    Per-episode scores come from test/sim_max_reward_<idx> keys (50 per eval line).
    When multiple lines share the same epoch, the last one wins.
    """
    import re
    ep_pattern = re.compile(r"^test/sim_max_reward_\d+$")
    results: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = _get(obj, SCORE_KEYS)
            if score is None:
                continue
            epoch = obj.get("epoch")
            if epoch is None:
                continue
            try:
                epoch = int(epoch)
            except (TypeError, ValueError):
                continue
            std = _get(obj, STD_KEYS)
            ep_scores = sorted(
                [float(v) * 100.0 for k, v in obj.items() if ep_pattern.match(k)]
            )
            results[epoch] = {
                "mean_score_pct":    float(score) * 100.0,
                "std_score_pct":     float(std) * 100.0 if std is not None else None,
                "per_episode_scores": ep_scores,
            }
    return results


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _aggregate(per_seed: list[dict[int, dict]]) -> tuple[list[EvalResult], list[float]]:
    """
    Aggregate across seeds. std_score = cross-seed std of per-epoch mean scores.
    Only includes epochs present in ALL seeds.

    Also returns the 50 per-episode scores from the single best (seed, epoch) —
    used for apples-to-apples statistical comparison against SODA's 50-episode evals.
    """
    all_epochs = set.intersection(*[set(d.keys()) for d in per_seed])
    common_epochs = sorted(e for e in all_epochs if e <= MAX_EPOCH)
    rows: list[EvalResult] = []
    best_ep_scores: list[float] = []
    best_mean = -1.0

    for epoch in common_epochs:
        seed_means = np.array([d[epoch]["mean_score_pct"] for d in per_seed])
        cross_std = float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0
        ep_scores_this_epoch = next(
            (d[epoch]["per_episode_scores"] for d in per_seed if d.get(epoch, {}).get("per_episode_scores")),
            [],
        )
        rows.append(EvalResult(
            epoch=epoch,
            checkpoint="",
            n_action_steps=8,
            duration_termination=True,
            n_episodes=len(ep_scores_this_epoch) or len(seed_means),
            mean_score=float(np.mean(seed_means)),
            std_score=cross_std,
            per_episode_scores=ep_scores_this_epoch,
            run_label=RUN_LABEL,
        ))
        # Track the best epoch by cross-seed mean (not by individual seed).
        # Per-episode scores come from the best single seed at that cross-seed-best epoch
        # so the 50 scores are internally consistent while epoch selection is unbiased.
        cross_mean = float(np.mean(seed_means))
        if cross_mean > best_mean:
            best_mean = cross_mean
            best_ep_scores = max(
                (d[epoch]["per_episode_scores"] for d in per_seed if d.get(epoch, {}).get("per_episode_scores")),
                key=lambda s: np.mean(s),
                default=[],
            )

    return rows, best_ep_scores


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _print_table(rows: list[EvalResult]) -> None:
    if not rows:
        print("No results to display.")
        return

    best = max(rows, key=lambda r: r.mean_score)
    report_set = {r.epoch for r in rows if r.epoch % 200 == 0}
    report_set.add(best.epoch)

    print(f"\n{'Epoch':>6}  {'Mean Score (%)':>14}  {'Cross-seed Std':>14}")
    print("-" * 40)
    for r in rows:
        if r.epoch in report_set:
            marker = "  ← best" if r.epoch == best.epoch else ""
            print(f"{r.epoch:>6}  {r.mean_score:>14.2f}  {r.std_score:>14.2f}{marker}")

    print(f"\nBest: epoch {best.epoch},  {best.mean_score:.2f}% ± {best.std_score:.2f}%  (cross-seed std, {len(SEEDS)} seeds)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot")
    args = parser.parse_args()

    print("=== Columbia Diffusion Policy (CNN) — Push-T Baseline ===\n")
    print("Downloading seed logs (cached after first run)...")

    per_seed: list[dict[int, dict]] = []
    for seed in SEEDS:
        path = _download_log(seed)
        print(f"  Parsing {seed}...", flush=True)
        data = _parse_log(path)
        if not data:
            print(f"  WARNING: no test/mean_score entries found in {seed}")
            continue
        epoch_range = f"{min(data)}–{max(data)}"
        print(f"    {len(data)} eval epochs  (range {epoch_range})")
        per_seed.append(data)

    if not per_seed:
        print("ERROR: no usable seed data found.")
        sys.exit(1)

    print(f"\nAggregating across {len(per_seed)} seeds...")
    rows, best_ep_scores = _aggregate(per_seed)
    print(f"  {len(rows)} epochs present in all seeds")
    print(f"  Best single checkpoint: {len(best_ep_scores)} per-episode scores extracted")

    # Save sweep results (cross-seed aggregated, for score-vs-epoch plots)
    save_sweep_results({RUN_LABEL: rows}, OUTPUT_JSON)

    # Also save best per-episode scores directly into the JSON for compare_best.py
    with open(OUTPUT_JSON) as f:
        out = json.load(f)
    out["best_per_episode_scores"] = best_ep_scores
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved best_per_episode_scores ({len(best_ep_scores)} episodes) → {OUTPUT_JSON}")

    _print_table(rows)

    if not args.no_plot:
        plot_eval_curve(
            {RUN_LABEL: rows},
            OUTPUT_PLOT,
            ylabel="Mean Max Overlap (%, t≤300)",
            title="Columbia Diffusion Policy (CNN) — Push-T Baseline",
        )
        print(f"Plot → {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
