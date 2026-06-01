"""
Eval standalone β network — termination_study stage 4.

Evaluates epoch_0050.ckpt and epoch_0100.ckpt via 50-episode hierarchical rollout.
Uses beta termination mode (open_loop=False, duration_termination=False).

The standalone β network is wired via rollout_hierarchical_external_beta:
  π_high: fixed high_starts_prev_opt checkpoint
  π_low:  k=9/epoch_0450.ckpt (actions only; internal β head bypassed)
  β:      standalone beta checkpoint from stage4 training

Checkpoint selection: highest mean rollout score across epochs [50, 100].
Note: beta_val_acc_pos/neg from training are informational only — rollout score is
the authoritative selection criterion.

Incremental: skips already-evaluated (epoch) pairs. Safe to re-run.

Usage (repo root):
  modal run experiments/final_experiments/pusht/termination_study/stage4_standalone_beta/modal_eval_stage4.py
  modal run --detach ...  (returns immediately; monitor in Modal dashboard)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import modal
import numpy as np

try:
    sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))
except IndexError:
    sys.path.insert(0, "/root/soda-policy/modal")

from modal_config import app, rollout_hierarchical_external_beta, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = (
    "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt"
)
LOW_CHECKPOINT = (
    "/experiments/final_experiments/pusht/kernel_size_study/k9/epoch_0450.ckpt"
)
CONFIG_PATH = "configs/pusht/obs_positive_negative_standalone.yaml"

STAGE4_DIR = "/experiments/final_experiments/pusht/termination_study/stage4/obs_positive_negative_separate"
N_EPISODES = 50
TEST_START_SEED = 100000
N_ACTION_STEPS = 8
BETA_TRANSITION = 0.92
N_WORST_VIDEOS = 5

OUTPUT_PATH = Path(
    "experiments/final_experiments/pusht/termination_study/stage4_standalone_beta/stage4_results.json"
)
VOLUME_OUTPUT_PATH = (
    f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/termination_study/stage4/stage4_results.json"
)

LOCAL_DEBUG_DIR = Path(
    "experiments/final_experiments/pusht/termination_study/stage4_standalone_beta/debug_videos/obs_positive_negative_separate"
)


def _load_existing(vol: modal.Volume) -> list[dict]:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        existing = d.get("results", [])
        print(f"Loaded existing results from volume: {len(existing)} epochs evaluated")
        return existing
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        existing = d.get("results", [])
        print(f"Loaded existing results from local: {len(existing)} epochs evaluated")
        return existing
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


def _find_epoch_checkpoints(vol: modal.Volume, epochs: list[int]) -> dict[int, str]:
    rel_dir = volume_relative_path(STAGE4_DIR)
    try:
        entries = vol.listdir(rel_dir)
    except Exception as e:
        print(f"  Warning: {STAGE4_DIR}: {e}")
        return {}
    found = {}
    for entry in entries:
        fname = Path(entry.path).name
        m = re.match(r"epoch_(\d{4})\.ckpt$", fname)
        if m and int(m.group(1)) in epochs:
            found[int(m.group(1))] = f"{STAGE4_DIR.rstrip('/')}/{fname}"
    return found


def _download_worst_videos(vol: modal.Volume, results: list[dict]) -> None:
    downloaded = 0
    for r in results:
        epoch = r["epoch"]
        scores = r.get("per_episode_scores", [])
        if not scores:
            continue
        indexed = sorted(enumerate(scores), key=lambda x: x[1])
        worst_indices = [i for i, _ in indexed[:N_WORST_VIDEOS]]

        vol_base = (
            f"final_experiments/pusht/termination_study/stage4"
            f"/debug_videos/obs_positive_negative_separate/epoch_{epoch:04d}"
        )
        local_base = LOCAL_DEBUG_DIR / f"epoch_{epoch:04d}"
        local_base.mkdir(parents=True, exist_ok=True)

        for ep_idx in worst_indices:
            seed = TEST_START_SEED + ep_idx
            fname = f"ep{ep_idx:04d}_seed{seed}.mp4"
            vol_path = volume_relative_path(f"/experiments/{vol_base}/{fname}")
            local_path = local_base / fname
            if local_path.exists():
                continue
            try:
                data = b"".join(vol.read_file(vol_path))
                local_path.write_bytes(data)
                print(f"  downloaded: {local_path.name} (score={scores[ep_idx]:.1f}%)")
                downloaded += 1
            except Exception:
                pass
    print(f"Downloaded {downloaded} worst-{N_WORST_VIDEOS} debug video(s) → {LOCAL_DEBUG_DIR.resolve()}")


def _print_summary(results: list[dict]) -> None:
    print("\n=== Stage 4 summary ===")
    if not results:
        print("  No results yet.")
        return
    best = max(results, key=lambda r: r["mean_score"])
    for r in sorted(results, key=lambda r: r["epoch"]):
        marker = " [BEST]" if r["epoch"] == best["epoch"] else ""
        print(f"  epoch={r['epoch']:4d}  mean={r['mean_score']:.4f} ± {r['std_score']:.4f}{marker}")
    print(f"\n  WINNER: epoch={best['epoch']}  mean={best['mean_score']:.4f}")
    print(f"  Checkpoint: {STAGE4_DIR}/epoch_{best['epoch']:04d}.ckpt")


@app.function(
    image=image,
    gpu=None,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_stage4(
    work_items: list[dict],
    existing: list[dict],
    n_action_steps: int,
) -> dict:
    """Spawn rollouts, collect results, save JSON to Modal Volume. Runs entirely on Modal."""
    from pathlib import Path

    results = list(existing)
    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

    calls = []
    for item in work_items:
        debug_video_dir = (
            f"/experiments/final_experiments/pusht/termination_study/stage4"
            f"/debug_videos/obs_positive_negative_separate/epoch_{item['epoch']:04d}"
        )
        call = rollout_hierarchical_external_beta.spawn(
            config_path=CONFIG_PATH,
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=LOW_CHECKPOINT,
            standalone_beta_checkpoint=item["ckpt_path"],
            n_episodes=N_EPISODES,
            test_start_seed=TEST_START_SEED,
            n_action_steps=n_action_steps,
            beta_transition=BETA_TRANSITION,
            max_steps=300,
            no_video=False,
            output_dir=debug_video_dir,
        )
        calls.append((item, call))

    print(f"Spawned {len(calls)} rollout jobs. Collecting results...")

    for item, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0

        step_means, step_stds = {}, {}
        if episodes:
            all_step_keys = {k for ep in episodes for k in ep["metrics"].get("max_overlap_at_step", {}).keys()}
            for t in sorted(all_step_keys, key=int):
                vals = [ep["metrics"].get("max_overlap_at_step", {}).get(t, 0.0) for ep in episodes]
                step_means[f"mean_score@{t}"] = float(np.mean(vals))
                step_stds[f"std_score@{t}"] = float(np.std(vals))

        results.append({
            "epoch": item["epoch"],
            "checkpoint": item["ckpt_path"],
            "n_action_steps": n_action_steps,
            "beta_transition": BETA_TRANSITION,
            "duration_termination": False,
            "n_episodes": len(scores),
            "mean_score": mean_score,
            "std_score": std_score,
            **step_means,
            **step_stds,
            "per_episode_scores": scores,
        })
        print(f"  epoch {item['epoch']}: mean={mean_score:.4f} ± {std_score:.4f}")

        results.sort(key=lambda r: r["epoch"])
        best = max(results, key=lambda r: r["mean_score"]) if results else None
        vol_path.write_text(json.dumps({
            "n_action_steps": n_action_steps,
            "beta_transition": BETA_TRANSITION,
            "results": results,
            "best": best,
        }, indent=2))
        volume.commit()

    best = max(results, key=lambda r: r["mean_score"]) if results else None
    output = {
        "n_action_steps": n_action_steps,
        "beta_transition": BETA_TRANSITION,
        "results": results,
        "best": best,
    }
    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    if best:
        print(f"WINNER: epoch={best['epoch']}  mean={best['mean_score']:.4f}")
    return output


@app.local_entrypoint()
def main(n_action_steps: int = N_ACTION_STEPS) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    eval_epochs = [50, 100]

    existing = _load_existing(vol)
    already_done = _already_evaluated(existing)

    epoch_ckpts = _find_epoch_checkpoints(vol, eval_epochs)
    new = {e: p for e, p in epoch_ckpts.items() if e not in already_done}
    skipped = len(epoch_ckpts) - len(new)
    print(f"Found epochs {sorted(epoch_ckpts.keys())}, {skipped} already done, {len(new)} new")

    work_items = [
        {"epoch": epoch, "ckpt_path": ckpt_path}
        for epoch, ckpt_path in sorted(new.items())
    ]

    if not work_items:
        print("No new checkpoints to evaluate.")
        _print_summary(existing)
        _download_from_volume(vol)
        _download_worst_videos(vol, existing)
        return

    print(f"\nSubmitting {len(work_items)} eval job(s) to Modal aggregator...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    agg_call = _eval_aggregate_stage4.spawn(work_items, existing, n_action_steps)
    result = agg_call.get()

    _print_summary(result["results"])
    _download_from_volume(vol)
    _download_worst_videos(vol, result["results"])
