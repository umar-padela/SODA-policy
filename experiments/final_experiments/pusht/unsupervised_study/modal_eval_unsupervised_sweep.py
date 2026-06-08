"""
Epoch sweep eval for E3 — Push-T LOVE unsupervised option discovery.

Incremental: loads existing unsupervised_results.json from Modal Volume (authoritative)
or local fallback, skips already-evaluated epochs. Safe to re-run.

Usage:
  modal run experiments/final_experiments/pusht/unsupervised_study/modal_eval_unsupervised_sweep.py
  modal run experiments/final_experiments/pusht/unsupervised_study/modal_eval_unsupervised_sweep.py --download-only
"""

from __future__ import annotations
import json
import re
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

HIGH_CHECKPOINT = (
    "/experiments/final_experiments/pusht/unsupervised_study/high/best.ckpt"
)
LOW_DIR = "/experiments/final_experiments/pusht/unsupervised_study/low"
LOW_CONFIG = "configs/pusht_unsupervised/unsupervised_k5_no_beta.yaml"

N_EPISODES = 50
TEST_START_SEED = 100000
EPOCH_STRIDE = 50
MIN_EPOCH = 50
N_ACTION_STEPS = 8

OUTPUT_PATH = Path("experiments/final_experiments/pusht/unsupervised_study/unsupervised_results.json")
VOLUME_OUTPUT_PATH = f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/unsupervised_study/unsupervised_results.json"


def _load_existing(vol: modal.Volume) -> list[dict]:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        results = d.get("results", [])
        print(f"Loaded {len(results)} existing result(s) from volume")
        return results
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        results = d.get("results", [])
        print(f"Loaded {len(results)} existing result(s) from local")
        return results
    return []


def _already_evaluated(existing: list[dict]) -> set[int]:
    return {r["epoch"] for r in existing}


def _download_from_volume(vol: modal.Volume) -> None:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_bytes(data)
        print(f"Downloaded results → {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Warning: could not download from volume: {e}")


def _discover_epoch_checkpoints(vol: modal.Volume) -> list[tuple[int, str]]:
    rel_dir = volume_relative_path(LOW_DIR)
    try:
        entries = vol.listdir(rel_dir)
    except Exception as e:
        print(f"  Warning: could not list {LOW_DIR}: {e}")
        return []
    results = []
    for entry in entries:
        fname = Path(entry.path).name
        m = re.match(r"epoch_(\d{4})\.ckpt$", fname)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch >= MIN_EPOCH and epoch % EPOCH_STRIDE == 0:
            results.append((epoch, f"{LOW_DIR.rstrip('/')}/{fname}"))
    return sorted(results)


def _print_summary(results: list[dict]) -> None:
    print("\n=== Unsupervised study summary ===")
    if not results:
        print("  No results yet")
        return
    best = max(results, key=lambda r: r["mean_score"])
    print(f"  Best: epoch={best['epoch']}, mean={best['mean_score']:.4f} ± {best['std_score']:.4f}")
    print(f"  All epochs: {sorted(r['epoch'] for r in results)}")


@app.function(
    image=image,
    gpu=None,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_unsupervised(
    work_items: list[dict],
    existing: list[dict],
) -> dict:
    """Spawn rollouts, collect results, save JSON to Modal Volume. Runs entirely on Modal."""
    import json
    import numpy as np
    from pathlib import Path

    calls = []
    for item in work_items:
        debug_video_dir = (
            f"/experiments/final_experiments/pusht/unsupervised_study"
            f"/debug_videos/epoch_{item['epoch']:04d}"
        )
        call = rollout_hierarchical.spawn(
            config_path=item["config"],
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=item["ckpt_path"],
            n_episodes=N_EPISODES,
            test_start_seed=TEST_START_SEED,
            n_action_steps=N_ACTION_STEPS,
            duration_termination=True,
            open_loop=False,
            max_steps=300,
            no_video=False,
            output_dir=debug_video_dir,
        )
        calls.append((item, call))

    print(f"Spawned {len(calls)} rollout job(s). Collecting results...")

    results = list(existing)
    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

    for item, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0

        step_means: dict[str, float] = {}
        step_stds: dict[str, float] = {}
        if episodes:
            all_step_keys = {
                k for ep in episodes
                for k in ep["metrics"].get("max_overlap_at_step", {}).keys()
            }
            for t in sorted(all_step_keys, key=int):
                vals = [ep["metrics"].get("max_overlap_at_step", {}).get(t, 0.0) for ep in episodes]
                step_means[f"mean_score@{t}"] = float(np.mean(vals))
                step_stds[f"std_score@{t}"] = float(np.std(vals))

        results.append({
            "epoch": item["epoch"],
            "checkpoint": item["ckpt_path"],
            "n_action_steps": N_ACTION_STEPS,
            "duration_termination": True,
            "n_episodes": len(scores),
            "mean_score": mean_score,
            "std_score": std_score,
            **step_means,
            **step_stds,
            "per_episode_scores": scores,
        })
        print(f"  epoch {item['epoch']:4d}: mean={mean_score:.4f}  std={std_score:.4f}")

        results.sort(key=lambda r: r["epoch"])
        vol_path.write_text(json.dumps({"results": results}, indent=2))
        volume.commit()

    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    return {"results": results}


@app.local_entrypoint()
def main(download_only: bool = False) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    existing = _load_existing(vol)

    if download_only:
        _print_summary(existing)
        _download_from_volume(vol)
        return

    already_done = _already_evaluated(existing)
    epoch_ckpts = _discover_epoch_checkpoints(vol)
    new = [(e, p) for e, p in epoch_ckpts if e not in already_done]
    skipped = len(epoch_ckpts) - len(new)
    print(f"Checkpoints found: {len(epoch_ckpts)}, already done: {skipped}, new: {len(new)}")

    if not new:
        print("\nNo new checkpoints to evaluate.")
        _print_summary(existing)
        _download_from_volume(vol)
        return

    work_items = [
        {"epoch": epoch, "ckpt_path": ckpt_path, "config": LOW_CONFIG}
        for epoch, ckpt_path in new
    ]

    print(f"\nSubmitting {len(work_items)} eval job(s) to Modal aggregator...")
    agg_call = _eval_aggregate_unsupervised.spawn(work_items, existing)
    result = agg_call.get()

    _print_summary(result["results"])
    _download_from_volume(vol)
