"""
Sweep n_action_steps on the best SODA policy from termination_study — receding_horizon_study.

Incremental: loads existing results from Modal Volume (authoritative) or local fallback,
skips already-evaluated n_action_steps values. Safe to re-run.

All collection and saving happens inside a Modal remote function, so the local terminal
can be closed after spawning. Use --detach to return immediately:

  modal run --detach experiments/final_experiments/pusht/receding_horizon_study/modal_eval_n_action_steps.py \\
    --low-checkpoint /experiments/... --config configs/pusht/...

If the terminal closes mid-run, re-running will find the completed results on the volume.
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

HIGH_CHECKPOINT = (
    "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt"
)
# Best policy from termination study: k=9 epoch 450, duration termination
DEFAULT_LOW_CHECKPOINT = "/experiments/final_experiments/pusht/kernel_size_study/k9/epoch_0450.ckpt"
DEFAULT_CONFIG = "configs/pusht/exp_k9_no_beta.yaml"

N_EPISODES = 50
TEST_START_SEED = 100000
N_ACTION_STEPS_VALUES = [2, 4, 6, 8, 12, 16]

OUTPUT_PATH = Path(
    "experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/sweep_results.json"
)
VOLUME_OUTPUT_PATH = (
    f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/sweep_results.json"
)
LOCAL_DEBUG_DIR = Path(
    "experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/debug_videos"
)


def _download_worst_videos(vol: modal.Volume, sweep_results: dict, n_worst: int = 5) -> None:
    # sweep_results is flat: {label: {per_episode_scores, ...}} — no epoch dimension
    downloaded = 0
    for label, r in sweep_results.items():
        scores = r.get("per_episode_scores", [])
        if not scores:
            continue
        worst = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1])[:n_worst]]
        vol_base = f"final_experiments/pusht/receding_horizon_study/debug_videos/{label}"
        out_dir = LOCAL_DEBUG_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for ep_idx in worst:
            seed = TEST_START_SEED + ep_idx
            fname = f"ep{ep_idx:04d}_seed{seed}.mp4"
            out_file = out_dir / fname
            if out_file.exists():
                continue
            try:
                data = b"".join(vol.read_file(volume_relative_path(f"/experiments/{vol_base}/{fname}")))
                out_file.write_bytes(data)
                print(f"  {label}/{fname}  ({scores[ep_idx]:.1f}%)")
                downloaded += 1
            except Exception:
                pass
    print(f"  {downloaded} video(s) → {LOCAL_DEBUG_DIR}")


def _run_plot(output_path: Path) -> None:
    plot_script = Path(__file__).parent / "plot_n_action_steps.py"
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
        print(f"Loaded existing results from volume: {list(d.keys())}")
        return d
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        print(f"Loaded existing results from local: {list(d.keys())}")
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


def _print_summary(sweep_results: dict) -> None:
    if not sweep_results:
        return
    print("\n=== n_action_steps sweep summary ===")
    print(f"{'Label':>12}  {'n_steps':>8}  {'Mean':>8}  {'Std':>6}")
    print("-" * 45)
    for label_key in [f"n{n}" for n in N_ACTION_STEPS_VALUES] + ["open_loop"]:
        r = sweep_results.get(label_key)
        if r is None:
            continue
        print(f"  {label_key:>10}  {str(r.get('n_action_steps', '-')):>8}  "
              f"{r.get('mean_score', 0):.4f}  {r.get('std_score', 0):.4f}")
    if sweep_results:
        best_label = max(sweep_results, key=lambda k: sweep_results[k]["mean_score"])
        print(f"\nBest: {best_label} → mean={sweep_results[best_label]['mean_score']:.4f}")


@app.function(
    image=image,
    gpu=None,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_n_action_steps(
    work_items: list[dict],
    existing: dict,
    low_checkpoint: str,
    config: str,
    n_episodes: int,
    max_concurrent: int = 4,
) -> dict:
    """Spawn rollouts in batches, collect results, save to volume after each."""
    import json
    import numpy as np
    from pathlib import Path

    sweep_results = dict(existing)

    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

    for batch_start in range(0, len(work_items), max_concurrent):
        batch = work_items[batch_start:batch_start + max_concurrent]
        calls = []
        for item in batch:
            debug_video_dir = (
                f"/experiments/final_experiments/pusht/receding_horizon_study"
                f"/debug_videos/{item['label']}"
            )
            if item["label"] == "open_loop":
                call = rollout_hierarchical.spawn(
                    config_path=config,
                    high_checkpoint=HIGH_CHECKPOINT,
                    low_checkpoint=low_checkpoint,
                    n_episodes=n_episodes,
                    test_start_seed=TEST_START_SEED,
                    n_action_steps=8,
                    duration_termination=False,
                    open_loop=True,
                    max_steps=300,
                    no_video=False,
                    output_dir=debug_video_dir,
                )
            else:
                call = rollout_hierarchical.spawn(
                    config_path=config,
                    high_checkpoint=HIGH_CHECKPOINT,
                    low_checkpoint=low_checkpoint,
                    n_episodes=n_episodes,
                    test_start_seed=TEST_START_SEED,
                    n_action_steps=item["n_action_steps"],
                    duration_termination=True,
                    open_loop=False,
                    max_steps=300,
                    no_video=False,
                    output_dir=debug_video_dir,
                )
            calls.append((item, call))

        print(f"Spawned batch of {len(calls)} rollout jobs (items {batch_start+1}–{batch_start+len(calls)} of {len(work_items)}). Collecting...")

        for item, call in calls:
            label = item["label"]
            try:
                result = call.get()
            except Exception as e:
                print(f"  {label:>10}: FAILED ({e}) — skipping")
                continue
            episodes = result.get("episodes") or []
            scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
            mean_score = float(np.mean(scores)) if scores else 0.0
            std_score = float(np.std(scores)) if scores else 0.0
            sweep_results[label] = {
                "n_action_steps": item.get("n_action_steps"),
                "label": label,
                "mean_score": mean_score,
                "std_score": std_score,
                "n_episodes": len(scores),
                "per_episode_scores": scores,
                "low_checkpoint": low_checkpoint,
            }
            print(f"  {label:>10}: mean={mean_score:.4f}  std={std_score:.4f}")

            # Save after every result — crash-safe
            vol_path.write_text(json.dumps(sweep_results, indent=2))
            volume.commit()

    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    return sweep_results


@app.local_entrypoint()
def main(
    low_checkpoint: str = DEFAULT_LOW_CHECKPOINT,
    config: str = DEFAULT_CONFIG,
    n_episodes: int = N_EPISODES,
    download_only: bool = False,
    max_concurrent: int = 4,
) -> None:
    """
    Sweep n_action_steps on the best policy from termination_study.

    Defaults to k=9 epoch 450 (duration termination) — the best overall policy.
    Override with --low-checkpoint and --config if evaluating a different checkpoint.

    Parameters
    ----------
    low_checkpoint
        Volume path to the low policy checkpoint.
    config
        Config yaml matching the checkpoint.
    n_episodes
        Episodes per eval (default 50).
    download_only
        Skip eval — just download latest results + worst videos and regenerate plots.
    """
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    sweep_results = _load_existing(vol)

    if download_only:
        _print_summary(sweep_results)
        _download_from_volume(vol)
        _download_worst_videos(vol, sweep_results)
        _run_plot(OUTPUT_PATH)
        return

    print(f"Sweeping n_action_steps on: {low_checkpoint}")
    print(f"Config: {config}")

    work_items = []
    for n in N_ACTION_STEPS_VALUES:
        label = f"n{n}"
        if label in sweep_results:
            print(f"  {label}: already evaluated, skipping")
        else:
            work_items.append({"label": label, "n_action_steps": n})

    if "open_loop" in sweep_results:
        print("  open_loop: already evaluated, skipping")
    else:
        work_items.append({"label": "open_loop", "n_action_steps": None})

    if not work_items:
        print("\nNo new configs to evaluate.")
        _print_summary(sweep_results)
        _download_from_volume(vol)
        _download_worst_videos(vol, sweep_results)
        _run_plot(OUTPUT_PATH)
        return

    print(f"\nSubmitting {len(work_items)} eval jobs to Modal aggregator...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    agg_call = _eval_aggregate_n_action_steps.spawn(work_items, sweep_results, low_checkpoint, config, n_episodes, max_concurrent)
    result = agg_call.get()

    _print_summary(result)
    _download_from_volume(vol)
    _download_worst_videos(vol, result)
    _run_plot(OUTPUT_PATH)
