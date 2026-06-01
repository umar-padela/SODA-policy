"""
Targeted video eval — run specific seeds with video recording enabled.

Use this to debug outlier episodes (0% scores, unexpected failures) from any
experiment without re-running the full eval sweep.

Videos are stored under the study that produced the checkpoint, organized as:
  experiments/final_experiments/pusht/{study}/debug_videos/{run}/{epoch}/

Examples
--------
  # kernel_size_study k=5 epoch 50:
  modal run experiments/final_experiments/pusht/shared/modal_eval_videos.py \\
    --study kernel_size_study --run k5 \\
    --low-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k5/epoch_0050.ckpt \\
    --config configs/pusht/exp_k5_no_beta.yaml \\
    --seeds 100026,100049

  # termination_study stage1 bottleneck_expert epoch 100:
  modal run experiments/final_experiments/pusht/shared/modal_eval_videos.py \\
    --study termination_study --run bottleneck_expert \\
    --low-checkpoint /experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert/epoch_0100.ckpt \\
    --config configs/pusht/exp_term_bottleneck_expert.yaml \\
    --seeds 100026,100049

  # Videos download locally by default. Pass --no-download to skip.

Output paths (volume / local):
  /experiments/final_experiments/pusht/{study}/debug_videos/{run}/{epoch}/
  experiments/final_experiments/pusht/{study}/debug_videos/{run}/{epoch}/
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))

import modal  # noqa: E402
from modal_config import app, rollout_hierarchical  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = (
    "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt"
)

PUSHT_FINAL_ROOT = "experiments/final_experiments/pusht"


def _epoch_label(checkpoint_path: str) -> str:
    """Derive a folder name from the checkpoint filename, e.g. epoch_0050 or best."""
    stem = Path(checkpoint_path).stem  # e.g. "epoch_0050" or "best"
    return stem


def _build_output_dir(study: str, run: str, epoch_label: str) -> tuple[str, Path]:
    """Return (volume_path, local_path) for debug videos."""
    subpath = f"{study}/debug_videos/{run}/{epoch_label}"
    volume_dir = f"/experiments/final_experiments/pusht/{subpath}"
    local_dir = Path(PUSHT_FINAL_ROOT) / subpath
    return volume_dir, local_dir


def _download_videos(volume_output_dir: str, local_dir: Path) -> None:
    """Download all mp4s from the volume debug dir to the local experiments folder."""
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    rel_dir = volume_relative_path(volume_output_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        entries = list(vol.listdir(rel_dir))
    except Exception as e:
        print(f"  Warning: could not list volume dir {rel_dir}: {e}")
        return

    mp4s = [e for e in entries if e.path.endswith(".mp4")]
    if not mp4s:
        print("  No mp4 files found on volume to download.")
        return

    print(f"\nDownloading {len(mp4s)} video(s) → {local_dir}/")
    for entry in mp4s:
        fname = Path(entry.path).name
        local_path = local_dir / fname
        data = b"".join(vol.read_file(entry.path))
        local_path.write_bytes(data)
        print(f"  {fname}  ({len(data) / 1_000_000:.1f} MB)")
    print(f"Done. Videos in {local_dir}/")


@app.local_entrypoint()
def main(
    study: str,
    run: str,
    low_checkpoint: str,
    config: str,
    seeds: str,
    duration_termination: bool = True,
    n_action_steps: int = 8,
    max_steps: int = 300,
    download: bool = True,
) -> None:
    """
    Parameters
    ----------
    study
        Experiment study name, e.g. kernel_size_study, termination_study,
        receding_horizon_study, comparison_study.
    run
        Run variant within the study, e.g. k5, k7, bottleneck_expert, n8.
    low_checkpoint
        Volume path to the checkpoint to debug.
    config
        Config yaml matching the checkpoint.
    seeds
        Comma-separated list of seeds, e.g. 100026,100049.
    duration_termination
        True for no-beta kernel_size_study runs; False for termination_study runs.
    n_action_steps
        Receding-horizon window (default 8).
    max_steps
        Episode length (default 300).
    download
        Download videos locally to experiments/final_experiments/pusht/{study}/debug_videos/{run}/{epoch}/.
        Pass --no-download to skip.
    """
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    epoch_label = _epoch_label(low_checkpoint)
    volume_output_dir, local_dir = _build_output_dir(study, run, epoch_label)

    print(f"Recording {len(seed_list)} episodes with video:")
    print(f"  study:      {study}/{run}/{epoch_label}")
    print(f"  checkpoint: {low_checkpoint}")
    print(f"  seeds:      {seed_list}")
    print(f"  volume:     {volume_output_dir}/")

    calls = []
    for seed in seed_list:
        # To hit a specific seed, set test_start_seed=seed and n_episodes=1.
        call = rollout_hierarchical.spawn(
            config_path=config,
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=low_checkpoint,
            n_episodes=1,
            test_start_seed=seed,
            n_action_steps=n_action_steps,
            duration_termination=duration_termination,
            open_loop=False,
            max_steps=max_steps,
            output_dir=volume_output_dir,
            no_video=False,
        )
        calls.append((seed, call))

    print(f"\nSpawned {len(calls)} jobs. Waiting...")
    for seed, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        if episodes:
            ep = episodes[0]
            score = ep["metrics"].get("max_overlap_full", 0.0)
            video = ep.get("video_path") or "no video"
            print(f"  seed={seed}: max_overlap={score:.1f}%  video={video}")

    if download:
        _download_videos(volume_output_dir, local_dir)
    else:
        rel = volume_relative_path(volume_output_dir)
        print(f"\nVideos on volume: {volume_output_dir}/")
        print("Re-run with --download to fetch them locally, or:")
        print(f"  modal volume get {MODAL_VOLUME_NAME} {rel}")
