"""
Statistical comparison: SODA k=5, SODA k=9 vs Diffusion Policy (CNN) baseline.

All comparisons are 50 episodes vs 50 episodes — same eval protocol, same seeds.

Data sources:
  SODA k=5, k=9 : kernel_sweep_results.json  (best epoch by mean_score, 50ep each)
  DP baseline   : final_comparison.json       (dp_baseline.per_episode_scores, 50ep)

Prerequisites:
  1. modal_eval_kernel_sweep.py must have run (SODA k=5 and k=9 results)
  2. modal_eval_final.py must have run with --dp-checkpoint (DP 50-episode eval)

Statistical tests (two-sample, two-tailed):
  Primary   — Mann-Whitney U (non-parametric; appropriate for bounded, non-normal scores)
  Secondary — Welch's t-test (parametric reference)

Usage:
  python experiments/final_experiments/pusht/comparison_study/compare_best.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE           = Path(__file__).parent
KERNEL_RESULTS = HERE.parent / "kernel_size_study" / "kernel_sweep_results.json"
DP_RESULTS     = HERE / "diffusion_policy_k5_results.json"   # Columbia logs fallback
DP_EVAL_LOG    = HERE / "dp_k5" / "eval_log.json"            # our eval (preferred)
OUTPUT_JSON    = HERE / "compare_best_results.json"

TIMESTEPS = [125, 150, 175, 200, 225, 250, 275, 300]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_kernel_best(kernel: str) -> dict:
    with open(KERNEL_RESULTS) as f:
        d = json.load(f)
    runs = d.get("kernels", {}).get(kernel, [])
    if not runs:
        raise ValueError(
            f"No results for kernel={kernel!r} in {KERNEL_RESULTS}. "
            "Run modal_eval_kernel_sweep.py first."
        )
    best = max(runs, key=lambda r: r["mean_score"])
    if not best.get("per_episode_scores"):
        raise ValueError(
            f"SODA {kernel} epoch {best['epoch']} has no per_episode_scores. "
            "Re-run modal_eval_kernel_sweep.py."
        )
    return best


def _load_dp_baseline() -> dict:
    """Load DP per-episode scores. Prefers our eval_log.json; falls back to Columbia logs."""
    if DP_EVAL_LOG.exists():
        d = json.loads(DP_EVAL_LOG.read_text())
        eps = d["metrics"].get("episodes", [])
        scores = [ep["max_overlap_full"] for ep in eps]
        horizon_means = {
            t: d["metrics"].get(f"mean_score@{t}", 0.0) * 100.0
            for t in TIMESTEPS
        }
        # Per-episode scores at each t cutoff (for Welch t-test at every horizon)
        horizon_episodes = {
            t: [ep["max_overlap_at_step"][str(t)] for ep in eps
                if str(t) in ep.get("max_overlap_at_step", {})]
            for t in TIMESTEPS
        }
        return {
            "per_episode_scores": scores,
            "epoch": "best",
            "horizon_means": horizon_means,
            "horizon_episodes": horizon_episodes,
            "source": "eval_log",
        }
    # Fallback: Columbia downloaded logs
    if not DP_RESULTS.exists():
        raise FileNotFoundError(
            f"Neither {DP_EVAL_LOG} nor {DP_RESULTS} found.\n"
            "Run modal_eval.py with --output-dir final_experiments/pusht/comparison_study/dp_k5"
        )
    with open(DP_RESULTS) as f:
        d = json.load(f)
    ep_scores = d.get("best_per_episode_scores", [])
    if not ep_scores:
        raise ValueError("best_per_episode_scores missing. Re-run fetch_dp_baseline.py.")
    rows = next((v for v in d.values() if isinstance(v, list)), [])
    best_epoch = max(rows, key=lambda r: r["mean_score"])["epoch"] if rows else "best"
    return {"per_episode_scores": ep_scores, "epoch": best_epoch, "horizon_means": {}, "source": "columbia_logs"}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _compare(a: list[float], b: list[float]) -> dict:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    mw_stat, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    t_stat,  t_p  = stats.ttest_ind(a, b, equal_var=False)
    return {
        "mannwhitney_u": float(mw_stat), "mannwhitney_p": float(mw_p),
        "mannwhitney_sig": _sig_stars(mw_p),
        "ttest_t": float(t_stat), "ttest_p": float(t_p),
        "ttest_sig": _sig_stars(t_p),
    }


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict]) -> None:
    W = 24
    header = (
        f"{'Policy':<{W}} {'Epoch':>6}  {'n':>3}  "
        f"{'Mean (%)':>9}  {'Std (%)':>8}  "
        f"{'MW p':>8}  {'t p':>8}  {'Sig':>3}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        mw_p = f"{r['mannwhitney_p']:.4f}" if r.get("mannwhitney_p") is not None else "—"
        t_p  = f"{r['ttest_p']:.4f}"       if r.get("ttest_p")       is not None else "—"
        sig  = r.get("mannwhitney_sig") or "—"
        print(
            f"{r['label']:<{W}} {str(r['epoch']):>6}  {r['n']:>3}  "
            f"{r['mean_pct']:>8.2f}%  {r['std_pct']:>7.2f}%  "
            f"{mw_p:>8}  {t_p:>8}  {sig:>3}"
        )
    print(sep)
    print("\nAll tests two-tailed vs DP baseline.")
    print("MW = Mann-Whitney U (primary)  |  t = Welch's t-test")
    print("Sig: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")


# ---------------------------------------------------------------------------
# Per-timestep table
# ---------------------------------------------------------------------------

def _ttest_from_stats(mean1: float, std1: float, n1: int,
                      mean2: float, std2: float, n2: int) -> str:
    """Welch's t-test from summary statistics. Returns significance stars."""
    _, p = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)
    return _sig_stars(p)


def _print_horizon_table(dp: dict, soda_k5: dict, soda_k9: dict,
                         dp_scores: list, k5_scores: list, k9_scores: list) -> None:
    """Table of mean scores + Welch t-test significance at every timestep cutoff.

    DP: per-episode scores at each t from eval_log.json.
    SODA: mean + std at each t from kernel_sweep_results.json.
    Mann-Whitney shown only at t=300 (requires raw per-episode for both sides).
    Welch's t-test shown at all t via ttest_ind_from_stats.
    """
    dp_horizons  = dp.get("horizon_means", {})          # t -> mean %
    dp_episodes  = dp.get("horizon_episodes", {})        # t -> list of 50 scores %
    k5_ep = soda_k5["epoch"]
    k9_ep = soda_k9["epoch"]
    N = 50

    W = 14
    header = (
        f"{'T':>5}  {'DP (k=5)':>{W}}  "
        f"{'SODA k=5 ep'+str(k5_ep):>{W}}  "
        f"{'SODA k=9 ep'+str(k9_ep):>{W}}  "
        f"{'k5 (t)':>7}  {'k9 (t)':>7}  {'k5 (MW)':>8}  {'k9 (MW)':>8}"
    )
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for t in TIMESTEPS:
        dp_m   = dp_horizons.get(t, float("nan"))
        dp_s   = float(np.std(dp_episodes.get(t, []))) if dp_episodes.get(t) else float("nan")
        k5_m   = soda_k5.get(f"mean@{t}", float("nan"))
        k5_s   = soda_k5.get(f"std@{t}",  float("nan"))
        k9_m   = soda_k9.get(f"mean@{t}", float("nan"))
        k9_s   = soda_k9.get(f"std@{t}",  float("nan"))

        # Welch t-test at every t (summary stats)
        k5_t = _ttest_from_stats(k5_m, k5_s, N, dp_m, dp_s, N) if not any(
            np.isnan(x) for x in [k5_m, k5_s, dp_m, dp_s]) else "—"
        k9_t = _ttest_from_stats(k9_m, k9_s, N, dp_m, dp_s, N) if not any(
            np.isnan(x) for x in [k9_m, k9_s, dp_m, dp_s]) else "—"

        # Mann-Whitney only at t=300 (raw per-episode available for both)
        if t == 300:
            k5_mw = _sig_stars(_compare(k5_scores, dp_scores)["mannwhitney_p"])
            k9_mw = _sig_stars(_compare(k9_scores, dp_scores)["mannwhitney_p"])
        else:
            k5_mw = k9_mw = "—"

        print(
            f"{t:>5}  {dp_m:>{W}.2f}%  {k5_m:>{W}.2f}%  {k9_m:>{W}.2f}%  "
            f"{k5_t:>7}  {k9_t:>7}  {k5_mw:>8}  {k9_mw:>8}"
        )
    print(sep)
    print("t = Welch t-test (all rows, summary stats)  |  MW = Mann-Whitney (t=300 only, raw scores)")
    print("Sig: *** p<0.001  ** p<0.01  * p<0.05  ns  — = not available")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== SODA vs Diffusion Policy — Statistical Comparison (50 vs 50 episodes) ===\n")

    dp      = _load_dp_baseline()
    soda_k5 = _load_kernel_best("k5")
    soda_k9 = _load_kernel_best("k9")

    print(f"DP data source: {dp['source']}")

    dp_scores  = dp["per_episode_scores"]
    k5_scores  = soda_k5["per_episode_scores"]
    k9_scores  = soda_k9["per_episode_scores"]

    # Attach per-t means and stds to soda dicts for the horizon table
    k_data = json.loads(KERNEL_RESULTS.read_text())
    for kernel, best in [("k5", soda_k5), ("k9", soda_k9)]:
        runs = k_data["kernels"][kernel]
        r = next(x for x in runs if x["epoch"] == best["epoch"])
        for t in TIMESTEPS:
            best[f"mean@{t}"] = r.get(f"mean_score@{t}", float("nan"))
            best[f"std@{t}"]  = r.get(f"std_score@{t}",  float("nan"))

    k5_test = _compare(k5_scores, dp_scores)
    k9_test = _compare(k9_scores, dp_scores)

    def _row(label, epoch, scores, test=None):
        return {
            "label": label,
            "epoch": epoch,
            "n": len(scores),
            "mean_pct": float(np.mean(scores)),
            "std_pct":  float(np.std(scores)),
            "mannwhitney_p": test["mannwhitney_p"] if test else None,
            "mannwhitney_sig": test["mannwhitney_sig"] if test else None,
            "ttest_p": test["ttest_p"] if test else None,
        }

    rows = [
        _row("Diffusion Policy (k=5)", dp["epoch"],      dp_scores),
        _row("SODA (k=5)",             soda_k5["epoch"], k5_scores, k5_test),
        _row("SODA (k=9)",             soda_k9["epoch"], k9_scores, k9_test),
    ]

    print("\n--- Overall comparison (t<=300) ---")
    _print_table(rows)

    print("\n--- Per-timestep mean scores ---")
    _print_horizon_table(dp, soda_k5, soda_k9, dp_scores, k5_scores, k9_scores)

    out = {
        "rows": rows,
        "tests": {"SODA (k=5) vs DP": k5_test, "SODA (k=9) vs DP": k9_test},
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
