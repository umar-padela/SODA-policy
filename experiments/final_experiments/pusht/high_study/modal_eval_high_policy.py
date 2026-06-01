"""
High policy study eval — all-frames high policy with k=9 ep450 low policy.

Duration termination is used (same as kernel_size_study) — this study is about
the high policy, not β.

Results saved to Modal Volume and downloaded locally. Safe to re-run.

Usage:
  modal run experiments/final_experiments/pusht/high_study/modal_eval_high_policy.py
  modal run experiments/final_experiments/pusht/high_study/modal_eval_high_policy.py --download-only
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import modal

try:
    sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))
except IndexError:
    sys.path.insert(0, "/root/soda-policy/modal")

from modal_config import app, rollout_hierarchical, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

LOW_CHECKPOINT  = "/experiments/final_experiments/pusht/kernel_size_study/k9/epoch_0450.ckpt"
HIGH_CHECKPOINT = "/experiments/final_experiments/pusht/high_study/high_all_frames/best.ckpt"
HIGH_CONFIG     = "configs/pusht/high_all_frames.yaml"

N_EPISODES      = 50
N_ACTION_STEPS  = 8
TEST_START_SEED = 100000

OUTPUT_PATH        = Path("experiments/final_experiments/pusht/high_study/high_policy_results.json")
VOLUME_OUTPUT_PATH = f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/high_study/high_policy_results.json"


def _load_existing(vol: modal.Volume) -> dict:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        existing = d.get("results", {})
        print(f"Loaded existing results from volume: {list(existing.keys())}")
        return existing
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        existing = d.get("results", {})
        print(f"Loaded existing results from local: {list(existing.keys())}")
        return existing
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


def _run_plot(output_path: Path) -> None:
    plot_script = Path(__file__).parent / "plot_high_policy.py"
    if not plot_script.exists():
        print(f"  (no plot script at {plot_script}, skipping)")
        return
    print(f"\nRunning {plot_script.name}...")
    subprocess.run([sys.executable, str(plot_script), "--data", str(output_path)], check=False)


@app.function(
    image=image,
    gpu=None,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_high(existing: dict) -> dict:
    """Spawn rollout, collect result, save JSON to Modal Volume. Runs entirely on Modal."""
    import json
    import numpy as np
    from pathlib import Path

    call = rollout_hierarchical.spawn(
        config_path=HIGH_CONFIG,
        high_checkpoint=HIGH_CHECKPOINT,
        low_checkpoint=LOW_CHECKPOINT,
        n_episodes=N_EPISODES,
        test_start_seed=TEST_START_SEED,
        n_action_steps=N_ACTION_STEPS,
        duration_termination=True,
        open_loop=False,
        max_steps=300,
        no_video=True,
    )

    print("Spawned rollout job. Waiting for result...")
    result = call.get()

    episodes = result.get("episodes") or []
    scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
    mean_score = float(np.mean(scores)) if scores else 0.0
    std_score  = float(np.std(scores))  if scores else 0.0

    step_means: dict[str, float] = {}
    step_stds:  dict[str, float] = {}
    if episodes:
        all_step_keys = {
            k for ep in episodes
            for k in ep["metrics"].get("max_overlap_at_step", {}).keys()
        }
        for t in sorted(all_step_keys, key=int):
            vals = [ep["metrics"].get("max_overlap_at_step", {}).get(t, 0.0) for ep in episodes]
            step_means[f"mean_score@{t}"] = float(np.mean(vals))
            step_stds[f"std_score@{t}"]   = float(np.std(vals))

    results = dict(existing)
    results["all_frames"] = {
        "label":                "High Policy (all frames)",
        "high_checkpoint":      HIGH_CHECKPOINT,
        "low_checkpoint":       LOW_CHECKPOINT,
        "n_action_steps":       N_ACTION_STEPS,
        "duration_termination": True,
        "n_episodes":           len(scores),
        "mean_score":           mean_score,
        "std_score":            std_score,
        "per_episode_scores":   scores,
        **step_means,
        **step_stds,
    }
    print(f"  all_frames: mean={mean_score:.4f}  std={std_score:.4f}")

    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)
    vol_path.write_text(json.dumps({"results": results}, indent=2))
    volume.commit()

    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    return {"results": results}


@app.local_entrypoint()
def main(download_only: bool = False) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    existing = _load_existing(vol)

    if download_only:
        _download_from_volume(vol)
        _run_plot(OUTPUT_PATH)
        return

    if "all_frames" in existing:
        print("all_frames already evaluated.")
        _download_from_volume(vol)
        _run_plot(OUTPUT_PATH)
        return

    print("Submitting eval job to Modal...")
    agg_call = _eval_aggregate_high.spawn(existing)
    result = agg_call.get()

    r = result["results"]["all_frames"]
    print(f"\nResult: mean={r['mean_score']:.4f}  std={r['std_score']:.4f}  n={r['n_episodes']}")

    _download_from_volume(vol)
    _run_plot(OUTPUT_PATH)
