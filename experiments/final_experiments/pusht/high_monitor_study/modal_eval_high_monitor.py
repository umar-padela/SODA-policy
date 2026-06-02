"""
High-policy-as-monitor experiment: π_high checks every step, switches option on disagreement.
No β head. Duration termination handles replanning within the current option.

Compares two π_low variants (k9-epoch350 vs k5-best) under the same π_high
(high_all_frames/best.ckpt).

The key design change vs old duration_termination:
  - OLD: π_high only called when duration says option_done → same-skill loop bug
  - NEW: π_high called every step; only clear_cache + switch on DISAGREEMENT
         Same-option prediction = keep executing (no kick-out, no loop)

Results saved as:
  high_monitor_study/
    k9_ep350_results.json
    k5_results.json

Usage
-----
modal run experiments/final_experiments/pusht/high_monitor_study/modal_eval_high_monitor.py \\
  --k9-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k9/epoch_0350.ckpt \\
  --k5-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k5/best.ckpt

Optional:
  --n-episodes 50
  --n-action-steps 8
  --skip-k9 / --skip-k5   run only one variant
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))

from modal_config import app, rollout_hierarchical, volume, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

# ---------------------------------------------------------------------------
# Study constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
VOLUME_STUDY_DIR = f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/high_monitor_study"

HIGH_CHECKPOINT = (
    "/experiments/final_experiments/pusht/high_study/high_all_frames/best.ckpt"
)
HIGH_CONFIG = "configs/pusht/high_all_frames.yaml"

DEFAULT_N_EPISODES = 50
DEFAULT_N_ACTION_STEPS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_filename(label: str) -> str:
    return f"{label}_results.json"


def _volume_result_path(label: str) -> str:
    return f"{VOLUME_STUDY_DIR}/{_result_filename(label)}"


def _local_result_path(label: str) -> Path:
    return HERE / _result_filename(label)


def _build_result(label: str, low_checkpoint: str, rollout_result: dict) -> dict:
    episodes = rollout_result.get("episodes", [])
    scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
    return {
        "label": label,
        "high_checkpoint": HIGH_CHECKPOINT,
        "low_checkpoint": low_checkpoint,
        "mode": "high_monitors_every_step",
        "n_episodes": len(scores),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "per_episode_scores": scores,
        "n_replans": rollout_result.get("episodes", [{}])[0].get("n_replans") if episodes else None,
    }


def _load_existing(vol: modal.Volume, label: str) -> dict | None:
    vol_rel = volume_relative_path(_volume_result_path(label))
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        if d.get("n_episodes", 0) > 0:
            return d
    except Exception:
        pass
    local = _local_result_path(label)
    if local.exists():
        d = json.loads(local.read_text())
        if d.get("n_episodes", 0) > 0:
            return d
    return None


def _save_to_volume(label: str, data: dict) -> None:
    vol_path = Path(_volume_result_path(label))
    vol_path.parent.mkdir(parents=True, exist_ok=True)
    vol_path.write_text(json.dumps(data, indent=2))
    volume.commit()
    print(f"  Saved to volume -> {vol_path}")


def _download_from_volume(vol: modal.Volume, label: str) -> bool:
    vol_rel = volume_relative_path(_volume_result_path(label))
    try:
        data = b"".join(vol.read_file(vol_rel))
        local = _local_result_path(label)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        print(f"  Downloaded {_result_filename(label)} -> {local}")
        return True
    except Exception as e:
        print(f"  Warning: could not download {_result_filename(label)}: {e}")
        return False


def _print_summary(results: dict[str, dict]) -> None:
    print("\n=== High Monitor Study Summary ===")
    print(f"{'Label':<20}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("-" * 46)
    for label, data in sorted(results.items()):
        if data:
            print(
                f"{label:<20}  {data['mean_score']:>7.1%}  "
                f"{data['std_score']:>7.1%}  {data['n_episodes']:>4}"
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
def _run_conditions(conditions: list[dict]) -> dict:
    """Spawn rollout_hierarchical calls in parallel, collect results, save JSONs."""
    import json
    import numpy as np
    from pathlib import Path

    calls = []
    for cond in conditions:
        label = cond["label"]
        out_dir = f"{VOLUME_STUDY_DIR}/{label}"
        call = rollout_hierarchical.spawn(
            config_path=HIGH_CONFIG,
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=cond["low_checkpoint"],
            n_episodes=cond["n_episodes"],
            n_action_steps=cond["n_action_steps"],
            duration_termination=True,
            high_monitors_every_step=True,
            no_video=False,
            video_failure_threshold=50.0,
            output_dir=out_dir,
        )
        calls.append((label, cond["low_checkpoint"], call))
        print(f"  Spawned: {label}")

    results = {}
    for label, low_ckpt, call in calls:
        try:
            rollout_result = call.get()
            data = _build_result(label, low_ckpt, rollout_result)
            results[label] = data
            vol_path = Path(_volume_result_path(label))
            vol_path.parent.mkdir(parents=True, exist_ok=True)
            vol_path.write_text(json.dumps(data, indent=2))
            volume.commit()
            print(
                f"  {label}: mean={data['mean_score']:.1%} +/- {data['std_score']:.1%} "
                f"(n={data['n_episodes']})"
            )
        except Exception as e:
            print(f"  {label}: FAILED ({e})")
            results[label] = {}

    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    k9_checkpoint: str = "/experiments/final_experiments/pusht/kernel_size_study/k9/epoch_0350.ckpt",
    k5_checkpoint: str = "/experiments/final_experiments/pusht/kernel_size_study/k5/best.ckpt",
    n_episodes: int = DEFAULT_N_EPISODES,
    n_action_steps: int = DEFAULT_N_ACTION_STEPS,
    skip_k9: bool = False,
    skip_k5: bool = False,
) -> None:
    """
    Run high-policy-as-monitor experiment for k9-epoch350 and k5 low policies.

    Parameters
    ----------
    k9_checkpoint
        Volume path to k9 epoch 350 checkpoint.
    k5_checkpoint
        Volume path to k5 best checkpoint.
    """
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    CONDITIONS = [
        {"label": "k9_ep350", "low_checkpoint": k9_checkpoint},
        {"label": "k5_best",  "low_checkpoint": k5_checkpoint},
    ]

    conditions = []
    all_results: dict[str, dict] = {}

    for cond in CONDITIONS:
        label = cond["label"]
        if (label == "k9_ep350" and skip_k9) or (label == "k5_best" and skip_k5):
            print(f"{label}: skipped by flag")
            continue
        existing = _load_existing(vol, label)
        if existing:
            print(f"{label}: already done (mean={existing['mean_score']:.1%}), skipping")
            all_results[label] = existing
        else:
            conditions.append({**cond, "n_episodes": n_episodes, "n_action_steps": n_action_steps})

    if conditions:
        print(f"\nSubmitting {len(conditions)} rollout jobs to Modal...")
        results = _run_conditions.remote(conditions)
        all_results.update(results)
    else:
        print("\nAll conditions already evaluated.")

    print("\nDownloading results...")
    for label in all_results:
        _download_from_volume(vol, label)

    _print_summary(all_results)
