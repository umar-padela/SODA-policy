"""
Final comparison: best SODA policy vs DP baseline — comparison_study.

Incremental: loads existing results from Modal Volume (authoritative) or local fallback.
Skips SODA eval if the same checkpoint is already saved; same for DP baseline.

All collection and saving happens inside a Modal remote function, so the local terminal
can be closed after spawning. Use --detach to return immediately:

  modal run --detach experiments/final_experiments/pusht/comparison_study/modal_eval_final.py \\
    --soda-checkpoint /experiments/... --dp-checkpoint /experiments/...

If the terminal closes mid-run, re-running will find the completed results on the volume.
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import modal

sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))

from modal_config import app, rollout_hierarchical, eval_run, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = (
    "/experiments/pusht/train_high_conditioned_prev_option/best.ckpt"
)
N_EPISODES = 50
TEST_START_SEED = 100000

OUTPUT_PATH = Path(
    "experiments/final_experiments/pusht/comparison_study/final_comparison.json"
)
VOLUME_OUTPUT_PATH = (
    f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/comparison_study/final_comparison.json"
)


def _run_plot(output_path: Path) -> None:
    plot_script = Path(__file__).parent / "plot_comparison.py"
    if not plot_script.exists():
        print(f"  (no plot script at {plot_script}, skipping)")
        return
    print(f"\nRunning {plot_script.name}...")
    subprocess.run([sys.executable, str(plot_script), "--data", str(output_path)], check=False)


def _load_existing(vol: modal.Volume) -> dict:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        existing_keys = [k for k in ("soda", "dp_baseline") if k in d]
        print(f"Loaded existing results from volume: {existing_keys}")
        return d
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        existing_keys = [k for k in ("soda", "dp_baseline") if k in d]
        print(f"Loaded existing results from local: {existing_keys}")
        return d
    return {}


def _download_from_volume(vol: modal.Volume) -> None:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_bytes(data)
        print(f"Downloaded results → {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Warning: could not download from volume: {e}")


def _print_summary(data: dict) -> None:
    soda = data.get("soda")
    dp = data.get("dp_baseline")
    if soda:
        print(f"  SODA:        {soda['mean_score']:.4f} ± {soda['std_score']:.4f}  (n={soda['n_episodes']})")
    if dp:
        print(f"  DP baseline: {dp['mean_score']:.4f}")
    if soda and dp:
        print(f"  SODA improvement: {(soda['mean_score'] - dp['mean_score']):+.4f}")


@app.function(
    image=image,
    gpu=None,
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_comparison(
    existing: dict,
    soda_checkpoint: str,
    dp_checkpoint: str,
    soda_config: str,
    dp_config: str,
    n_action_steps: int,
    run_soda: bool,
    run_dp: bool,
) -> dict:
    """Spawn SODA and DP evals, collect results, save JSON to Modal Volume."""
    import json
    import numpy as np
    from pathlib import Path

    soda_call = None
    dp_call = None

    if run_soda:
        soda_call = rollout_hierarchical.spawn(
            config_path=soda_config,
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=soda_checkpoint,
            n_episodes=N_EPISODES,
            test_start_seed=TEST_START_SEED,
            n_action_steps=n_action_steps,
            duration_termination=False,
            open_loop=False,
            max_steps=300,
            no_video=False,
        )

    if run_dp:
        dp_call = eval_run.spawn(
            config_path=dp_config,
            run_readme="comparison_study dp_baseline",
            checkpoint_path=dp_checkpoint,
            n_test=N_EPISODES,
            max_steps=300,
            n_action_steps=n_action_steps,
        )

    print(f"Spawned {int(run_soda) + int(run_dp)} eval jobs. Collecting results...")

    result = dict(existing)

    if soda_call is not None:
        soda_result = soda_call.get()
        soda_episodes = soda_result.get("episodes") or []
        soda_scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in soda_episodes]
        result["soda"] = {
            "checkpoint": soda_checkpoint,
            "config": soda_config,
            "n_action_steps": n_action_steps,
            "n_episodes": len(soda_scores),
            "mean_score": float(np.mean(soda_scores)) if soda_scores else 0.0,
            "std_score": float(np.std(soda_scores)) if soda_scores else 0.0,
            "per_episode_scores": soda_scores,
        }
        print(f"  SODA: mean={result['soda']['mean_score']:.4f} ± {result['soda']['std_score']:.4f}")

    if dp_call is not None:
        dp_result = dp_call.get()
        dp_metrics = dp_result.get("metrics", {})
        dp_mean = float(dp_metrics.get("mean_score@300", dp_metrics.get("mean_score", 0.0)))
        result["dp_baseline"] = {
            "checkpoint": dp_checkpoint,
            "config": dp_config,
            "n_episodes": N_EPISODES,
            "mean_score": dp_mean,
            "std_score": 0.0,
            "metrics": dp_metrics,
        }
        print(f"  DP baseline: mean={dp_mean:.4f}")

    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)
    vol_path.write_text(json.dumps(result, indent=2))
    volume.commit()
    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")

    return result


@app.local_entrypoint()
def main(
    soda_checkpoint: str,
    dp_checkpoint: str,
    soda_config: str = "configs/pusht/exp_term_obs_joint.yaml",
    dp_config: str = "configs/pusht/dp_frozen.yaml",
    n_action_steps: int = 8,
) -> None:
    """
    Run 50-episode eval for best SODA policy and DP baseline in parallel.

    Parameters
    ----------
    soda_checkpoint
        Volume path to the best SODA low policy checkpoint.
    dp_checkpoint
        Volume path to the DP baseline checkpoint.
    soda_config
        Config yaml for the SODA policy.
    dp_config
        Config yaml for DP baseline eval.
    n_action_steps
        Best value from receding_horizon_study.
    """
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    existing = _load_existing(vol)

    run_soda = "soda" not in existing or existing["soda"].get("checkpoint") != soda_checkpoint
    run_dp = "dp_baseline" not in existing or existing["dp_baseline"].get("checkpoint") != dp_checkpoint

    if not run_soda:
        print(f"SODA: already evaluated ({soda_checkpoint}), skipping")
    if not run_dp:
        print(f"DP: already evaluated ({dp_checkpoint}), skipping")

    if not run_soda and not run_dp:
        print("\nBoth already evaluated.")
        _print_summary(existing)
        _download_from_volume(vol)
        _run_plot(OUTPUT_PATH)
        return

    print(f"\nSubmitting {int(run_soda) + int(run_dp)} eval jobs to Modal aggregator...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    agg_call = _eval_aggregate_comparison.spawn(
        existing, soda_checkpoint, dp_checkpoint,
        soda_config, dp_config, n_action_steps,
        run_soda, run_dp,
    )
    result = agg_call.get()

    print("\n=== Final Comparison ===")
    _print_summary(result)
    _download_from_volume(vol)
    _run_plot(OUTPUT_PATH)
