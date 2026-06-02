"""
Noise robustness study: SODA vs DP baseline under action-space noise.

Sweeps noise levels η ∈ {0.0, 1.0, 1.5} for both DP and SODA, saves one JSON per
(policy, η) pair:

  noise_study/
    dp_eta0.0.json
    dp_eta1.0.json
    dp_eta1.5.json
    soda_eta0.0.json
    soda_eta1.0.json
    soda_eta1.5.json

Each JSON is labelled with noise_eta inside so plotting scripts can read it directly.

Noise model: AR(1) temporally correlated Gaussian noise added to executed actions
before env.step(), matching Liu et al. (2024) / BID paper.

Usage
-----
modal run experiments/final_experiments/pusht/noise_study/modal_eval_noise_study.py \\
  --dp-checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/best.ckpt \\
  --soda-low-checkpoint /experiments/train_low/soda_supervised/best.ckpt \\
  --soda-high-checkpoint /experiments/train_high/soda_supervised/best.ckpt

Optional flags:
  --noise-levels "0.0,1.0,1.5"          (comma-separated floats, default above)
  --n-episodes 50
  --n-action-steps 8
  --soda-config configs/pusht/soda_supervised.yaml
  --dp-config    configs/pusht/dp_frozen.yaml
  --skip-dp      run SODA only
  --skip-soda    run DP only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))

from modal_config import app, eval_run, volume, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

# ---------------------------------------------------------------------------
# Study constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
VOLUME_STUDY_DIR = f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/noise_study"

DEFAULT_NOISE_LEVELS = [0.0, 1.0, 1.5]
DEFAULT_N_EPISODES = 50
DEFAULT_N_ACTION_STEPS = 8
DEFAULT_DP_CONFIG = "configs/pusht/dp_frozen.yaml"
DEFAULT_SODA_CONFIG = "configs/pusht/soda_supervised.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_filename(policy: str, eta: float) -> str:
    return f"{policy}_eta{eta}.json"


def _volume_result_path(policy: str, eta: float) -> str:
    return f"{VOLUME_STUDY_DIR}/{_result_filename(policy, eta)}"


def _local_result_path(policy: str, eta: float) -> Path:
    return HERE / _result_filename(policy, eta)


def _build_result(policy: str, eta: float, rho: float, eval_result: dict) -> dict:
    """Extract per-episode scores and summary from eval_run result."""
    metrics = eval_result.get("metrics", {})
    episodes = metrics.get("episodes", [])
    per_episode_scores = [ep["max_overlap_full"] for ep in episodes]
    return {
        "policy": policy,
        "noise_eta": eta,
        "noise_rho": rho,
        "n_action_steps": eval_result.get("config", {}).get("n_action_steps", DEFAULT_N_ACTION_STEPS),
        "n_episodes": len(per_episode_scores),
        "mean_score": float(np.mean(per_episode_scores)) if per_episode_scores else 0.0,
        "std_score": float(np.std(per_episode_scores)) if per_episode_scores else 0.0,
        "per_episode_scores": per_episode_scores,
        "checkpoint": eval_result.get("checkpoint", ""),
        "config": eval_result.get("config", {}),
    }


def _load_existing(vol: modal.Volume, policy: str, eta: float) -> dict | None:
    vol_rel = volume_relative_path(_volume_result_path(policy, eta))
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        if d.get("n_episodes", 0) > 0:
            return d
    except Exception:
        pass
    local = _local_result_path(policy, eta)
    if local.exists():
        d = json.loads(local.read_text())
        if d.get("n_episodes", 0) > 0:
            return d
    return None


def _save_to_volume(vol: modal.Volume, policy: str, eta: float, data: dict) -> None:
    vol_path = Path(_volume_result_path(policy, eta))
    vol_path.parent.mkdir(parents=True, exist_ok=True)
    vol_path.write_text(json.dumps(data, indent=2))
    volume.commit()
    print(f"  Saved to volume → {vol_path}")


def _download_from_volume(vol: modal.Volume, policy: str, eta: float) -> bool:
    vol_rel = volume_relative_path(_volume_result_path(policy, eta))
    try:
        data = b"".join(vol.read_file(vol_rel))
        local = _local_result_path(policy, eta)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        print(f"  Downloaded {_result_filename(policy, eta)} → {local}")
        return True
    except Exception as e:
        print(f"  Warning: could not download {_result_filename(policy, eta)}: {e}")
        return False


def _print_summary(results: dict[tuple[str, float], dict]) -> None:
    print("\n=== Noise Study Summary ===")
    print(f"{'Policy':<8}  {'η':>5}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("─" * 42)
    for (policy, eta), data in sorted(results.items()):
        if data:
            print(
                f"{policy:<8}  {eta:>5.1f}  "
                f"{data['mean_score']:>7.1%}  {data['std_score']:>7.1%}  "
                f"{data['n_episodes']:>4}"
            )


# ---------------------------------------------------------------------------
# Modal aggregator
# ---------------------------------------------------------------------------

@app.function(
    image=__import__("modal_config", fromlist=["image"]).image,
    gpu=None,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _run_noise_sweep(
    conditions: list[dict],
    noise_rho: float = 0.9,
) -> dict:
    """
    Spawn all (policy, η) eval jobs in parallel, collect results, save JSONs.

    Each entry in ``conditions`` is::

        {"policy": "dp"|"soda", "eta": float, "config": str, "checkpoint": str,
         "high_checkpoint": str|None, "low_checkpoint": str|None,
         "n_episodes": int, "n_action_steps": int, "run_readme": str}
    """
    import json
    import numpy as np
    from pathlib import Path

    calls = []
    for cond in conditions:
        policy = cond["policy"]
        eta = cond["eta"]
        readme = cond.get("run_readme", f"noise_study {policy} eta={eta}")
        call = eval_run.spawn(
            config_path=cond["config"],
            run_readme=readme,
            checkpoint_path=cond.get("checkpoint"),
            high_checkpoint=cond.get("high_checkpoint"),
            low_checkpoint=cond.get("low_checkpoint"),
            n_test=cond.get("n_episodes", DEFAULT_N_EPISODES),
            n_action_steps=cond.get("n_action_steps", DEFAULT_N_ACTION_STEPS),
            record_video=False,
            noise_eta=eta,
            noise_rho=noise_rho,
        )
        calls.append((policy, eta, call))
        print(f"  Spawned: {policy} η={eta}")

    results: dict[tuple[str, float], dict] = {}
    for policy, eta, call in calls:
        try:
            eval_result = call.get()
            result = _build_result(policy, eta, noise_rho, eval_result)
            results[(policy, eta)] = result
            vol_path = Path(_volume_result_path(policy, eta))
            vol_path.parent.mkdir(parents=True, exist_ok=True)
            vol_path.write_text(json.dumps(result, indent=2))
            volume.commit()
            print(
                f"  {policy} η={eta}: "
                f"mean={result['mean_score']:.1%} ± {result['std_score']:.1%} "
                f"(n={result['n_episodes']})"
            )
        except Exception as e:
            print(f"  {policy} η={eta}: FAILED ({e})")
            results[(policy, eta)] = {}

    return {f"{policy}_eta{eta}": v for (policy, eta), v in results.items()}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    dp_checkpoint: str,
    soda_low_checkpoint: str,
    soda_high_checkpoint: str,
    dp_config: str = DEFAULT_DP_CONFIG,
    soda_config: str = DEFAULT_SODA_CONFIG,
    noise_levels: str = "0.0,1.0,1.5",
    noise_rho: float = 0.9,
    n_episodes: int = DEFAULT_N_EPISODES,
    n_action_steps: int = DEFAULT_N_ACTION_STEPS,
    skip_dp: bool = False,
    skip_soda: bool = False,
) -> None:
    """
    Run the noise robustness study for DP and SODA across noise levels.

    Parameters
    ----------
    dp_checkpoint
        Volume path to the DP baseline checkpoint.
    soda_low_checkpoint
        Volume path to the SODA π_low checkpoint.
    soda_high_checkpoint
        Volume path to the SODA π_high checkpoint.
    noise_levels
        Comma-separated η values (default "0.0,1.0,1.5").
    """
    etas = [float(x.strip()) for x in noise_levels.split(",")]
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    # Check what's already done
    conditions = []
    skipped = []
    for eta in etas:
        if not skip_dp:
            existing = _load_existing(vol, "dp", eta)
            if existing:
                skipped.append(("dp", eta, existing))
                print(f"DP η={eta}: already done (mean={existing['mean_score']:.1%}), skipping")
            else:
                conditions.append({
                    "policy": "dp",
                    "eta": eta,
                    "config": dp_config,
                    "checkpoint": dp_checkpoint,
                    "n_episodes": n_episodes,
                    "n_action_steps": n_action_steps,
                    "run_readme": f"noise_study dp eta={eta}",
                })
        if not skip_soda:
            existing = _load_existing(vol, "soda", eta)
            if existing:
                skipped.append(("soda", eta, existing))
                print(f"SODA η={eta}: already done (mean={existing['mean_score']:.1%}), skipping")
            else:
                conditions.append({
                    "policy": "soda",
                    "eta": eta,
                    "config": soda_config,
                    "high_checkpoint": soda_high_checkpoint,
                    "low_checkpoint": soda_low_checkpoint,
                    "n_episodes": n_episodes,
                    "n_action_steps": n_action_steps,
                    "run_readme": f"noise_study soda eta={eta}",
                })

    all_results: dict[tuple[str, float], dict] = {
        (p, e): d for p, e, d in skipped
    }

    if conditions:
        print(f"\nSubmitting {len(conditions)} eval jobs to Modal...")
        sweep_result = _run_noise_sweep.remote(conditions, noise_rho=noise_rho)
        for key, val in sweep_result.items():
            # key is "dp_eta1.0" → parse back
            parts = key.split("_eta")
            if len(parts) == 2:
                policy, eta_str = parts[0], parts[1]
                all_results[(policy, float(eta_str))] = val
    else:
        print("\nAll conditions already evaluated.")

    # Download all results locally
    print("\nDownloading results...")
    for (policy, eta) in all_results:
        _download_from_volume(vol, policy, eta)

    _print_summary(all_results)

    # Optionally run the plot script
    import subprocess
    plot_script = HERE / "plot_noise_study.py"
    if plot_script.exists():
        print(f"\nRunning {plot_script.name}...")
        subprocess.run([sys.executable, str(plot_script)], check=False)
