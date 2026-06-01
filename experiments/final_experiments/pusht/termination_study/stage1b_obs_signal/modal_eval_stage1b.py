"""
Eval obs_positive and obs_positive_negative — termination_study stage 1b eval.

Incremental: loads existing results from Modal Volume (authoritative) or local fallback,
skips already-evaluated (run, epoch) pairs. Safe to re-run.

All collection and saving happens inside a Modal remote function, so the local terminal
can be closed after spawning. Use --detach to return immediately:

  modal run --detach experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/modal_eval_stage1b.py

If the terminal closes mid-run, re-running will find the completed results on the volume.
"""

from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import modal

try:
    sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))
except IndexError:
    sys.path.insert(0, "/root/soda-policy/modal")

from modal_config import app, rollout_hierarchical, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = (
    "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt"
)
N_EPISODES = 50
TEST_START_SEED = 100000

OUTPUT_PATH = Path(
    "experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/stage1b_results.json"
)
VOLUME_OUTPUT_PATH = (
    f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/termination_study/stage1b/stage1b_results.json"
)

STAGE1B_RUNS = {
    "obs_positive": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage1b/obs_positive",
        "config": "configs/pusht/exp_term_obs_positive.yaml",
    },
    "obs_positive_negative": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage1b/obs_positive_negative",
        "config": "configs/pusht/exp_term_obs_positive_negative.yaml",
    },
    "bottleneck_ddim_positive_negative": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage1b/bottleneck_ddim_positive_negative",
        "config": "configs/pusht/exp_term_bottleneck_ddim_positive_negative.yaml",
    },
}


LOCAL_DEBUG_DIR = Path(
    "experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/debug_videos"
)
N_WORST_VIDEOS = 5


def _download_worst_videos(vol: modal.Volume, results: dict[str, list[dict]]) -> None:
    """Download the N_WORST_VIDEOS worst-scoring debug videos per (run, epoch) locally.

    Video filenames are deterministic: ep{i:04d}_seed{TEST_START_SEED+i}.mp4
    Only downloads files that actually exist on the volume (score > threshold means no video saved).
    """
    downloaded = 0
    for run_label, run_results in results.items():
        for r in run_results:
            epoch = r["epoch"]
            scores = r.get("per_episode_scores", [])
            if not scores:
                continue

            # Find indices of worst N scores
            indexed = sorted(enumerate(scores), key=lambda x: x[1])
            worst_indices = [i for i, _ in indexed[:N_WORST_VIDEOS]]

            vol_base = (
                f"final_experiments/pusht/termination_study/stage1b"
                f"/debug_videos/{run_label}/epoch_{epoch:04d}"
            )
            local_base = LOCAL_DEBUG_DIR / run_label / f"epoch_{epoch:04d}"
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
                    score = scores[ep_idx]
                    print(f"  downloaded: {local_path.name} (score={score:.1f}%)")
                    downloaded += 1
                except Exception:
                    pass  # video not saved (score was above threshold)

    print(f"Downloaded {downloaded} worst-{N_WORST_VIDEOS} debug video(s) → {LOCAL_DEBUG_DIR.resolve()}")


def _run_plot(output_path: Path) -> None:
    plot_script = Path(__file__).parent / "plot_stage1b.py"
    if not plot_script.exists():
        print(f"  (no plot script at {plot_script}, skipping)")
        return
    print(f"\nRunning {plot_script.name}...")
    subprocess.run([sys.executable, str(plot_script), "--data", str(output_path)], check=False)


def _load_existing(vol: modal.Volume) -> dict[str, list[dict]]:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        existing = d.get("results", {})
        print(f"Loaded existing results from volume: { {k: len(v) for k, v in existing.items()} }")
        return existing
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        existing = d.get("results", {})
        print(f"Loaded existing results from local: { {k: len(v) for k, v in existing.items()} }")
        return existing
    return {k: [] for k in STAGE1B_RUNS}


def _already_evaluated(existing: dict[str, list[dict]]) -> set[tuple[str, int]]:
    done = set()
    for run_label, results in existing.items():
        for r in results:
            done.add((run_label, r["epoch"]))
    return done


def _download_from_volume(vol: modal.Volume) -> None:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_bytes(data)
        print(f"Downloaded results → {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Warning: could not download from volume: {e}")


def _find_epoch_checkpoints(vol: modal.Volume, low_dir: str, epochs: list[int]) -> dict[int, str]:
    rel_dir = volume_relative_path(low_dir)
    try:
        entries = vol.listdir(rel_dir)
    except Exception as e:
        print(f"  Warning: {low_dir}: {e}")
        return {}
    found = {}
    for entry in entries:
        fname = Path(entry.path).name
        m = re.match(r"epoch_(\d{4})\.ckpt$", fname)
        if m and int(m.group(1)) in epochs:
            found[int(m.group(1))] = f"{low_dir.rstrip('/')}/{fname}"
    return found


def _print_summary(results: dict[str, list[dict]]) -> None:
    print("\n=== Stage 1b summary ===")
    for run_label, run_results in results.items():
        if not run_results:
            print(f"  {run_label}: no results yet")
            continue
        best = max(run_results, key=lambda r: r["mean_score"])
        print(f"  {run_label}: best epoch={best['epoch']}, mean={best['mean_score']:.4f} ± {best['std_score']:.4f}  ({len(run_results)} epochs evaluated)")


@app.function(
    image=image,
    gpu=None,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_stage1b(
    work_items: list[dict],
    existing: dict[str, list[dict]],
    n_action_steps: int,
) -> dict:
    """Spawn rollouts, collect results, save JSON to Modal Volume. Runs entirely on Modal."""
    import json
    import numpy as np
    from pathlib import Path

    calls = []
    for item in work_items:
        debug_video_dir = (
            f"/experiments/final_experiments/pusht/termination_study/stage1b"
            f"/debug_videos/{item['run_label']}/epoch_{item['epoch']:04d}"
        )
        call = rollout_hierarchical.spawn(
            config_path=item["config"],
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=item["ckpt_path"],
            n_episodes=N_EPISODES,
            test_start_seed=TEST_START_SEED,
            n_action_steps=n_action_steps,
            duration_termination=False,
            open_loop=False,
            max_steps=300,
            no_video=False,
            output_dir=debug_video_dir,
        )
        calls.append((item, call))

    print(f"Spawned {len(calls)} rollout jobs. Collecting results...")

    stage1b_results = {k: list(v) for k, v in existing.items()}

    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

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

        run_label = item["run_label"]
        epoch = item["epoch"]
        stage1b_results.setdefault(run_label, []).append({
            "epoch": epoch,
            "checkpoint": item["ckpt_path"],
            "n_action_steps": n_action_steps,
            "duration_termination": False,
            "n_episodes": len(scores),
            "mean_score": mean_score,
            "std_score": std_score,
            **step_means,
            **step_stds,
            "per_episode_scores": scores,
        })
        print(f"  {run_label} epoch {epoch}: mean={mean_score:.4f} ± {std_score:.4f}")

        # Save after every result — crash-safe
        for k in stage1b_results:
            stage1b_results[k].sort(key=lambda r: r["epoch"])
        best_per_run = {k: max(v, key=lambda r: r["mean_score"]) for k, v in stage1b_results.items() if v}
        winner = max(best_per_run, key=lambda k: best_per_run[k]["mean_score"]) if best_per_run else None
        vol_path.write_text(json.dumps({
            "n_action_steps": n_action_steps,
            "results": stage1b_results,
            "best_per_run": best_per_run,
            "winner": winner,
        }, indent=2))
        volume.commit()

    best_per_run = {k: max(v, key=lambda r: r["mean_score"]) for k, v in stage1b_results.items() if v}
    winner = max(best_per_run, key=lambda k: best_per_run[k]["mean_score"]) if best_per_run else None
    output = {"n_action_steps": n_action_steps, "results": stage1b_results, "best_per_run": best_per_run, "winner": winner}
    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    if winner:
        print(f"WINNER: {winner}")
    return output


@app.local_entrypoint()
def main(n_action_steps: int = 8) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    eval_epochs = [50, 100]

    stage1b_results = _load_existing(vol)
    already_done = _already_evaluated(stage1b_results)

    work_items = []
    for run_label, run_info in STAGE1B_RUNS.items():
        epoch_ckpts = _find_epoch_checkpoints(vol, run_info["low_dir"], eval_epochs)
        new = {e: p for e, p in epoch_ckpts.items() if (run_label, e) not in already_done}
        skipped = len(epoch_ckpts) - len(new)
        print(f"{run_label}: found epochs {sorted(epoch_ckpts.keys())}, {skipped} already done, {len(new)} new")
        for epoch, ckpt_path in sorted(new.items()):
            work_items.append({
                "run_label": run_label,
                "epoch": epoch,
                "ckpt_path": ckpt_path,
                "config": run_info["config"],
            })

    if not work_items:
        print("No new checkpoints to evaluate.")
        _print_summary(stage1b_results)
        _download_from_volume(vol)
        _download_worst_videos(vol, stage1b_results)
        _run_plot(OUTPUT_PATH)
        return

    print(f"\nSubmitting {len(work_items)} eval jobs to Modal aggregator...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    agg_call = _eval_aggregate_stage1b.spawn(work_items, stage1b_results, n_action_steps)
    result = agg_call.get()

    _print_summary(result["results"])
    if result.get("winner"):
        print(f"\nWINNER: {result['winner']}")
    _download_from_volume(vol)
    _download_worst_videos(vol, result["results"])
    _run_plot(OUTPUT_PATH)
