"""
Hierarchical rollout with standalone β checkpoint (stage4 / external beta).

The internal β head in the LowPolicy is bypassed; the standalone ResNet β model
evaluates termination at every step.

Examples (repo root):

  modal run modal/modal_rollout_external_beta.py \\
    --run-readme "stage4 debug" \\
    --standalone-beta-checkpoint /experiments/final_experiments/pusht/termination_study/stage4/obs_positive_negative_separate/epoch_0100.ckpt \\
    --local-save-path tmp/stage4_debug

  # More episodes, all videos saved
  modal run modal/modal_rollout_external_beta.py \\
    --run-readme "stage4 full qual" \\
    --standalone-beta-checkpoint /experiments/.../best.ckpt \\
    --n-episodes 10 --video-failure-threshold 100.0 \\
    --local-save-path tmp/stage4_qual
"""

from __future__ import annotations

import modal
from pathlib import Path

from modal_config import app, rollout_hierarchical_external_beta, spawn_modal_function
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path

HIGH_CHECKPOINT = "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt"
LOW_CHECKPOINT  = "/experiments/final_experiments/pusht/kernel_size_study/k9/epoch_0450.ckpt"
CONFIG_PATH     = "configs/pusht/obs_positive_negative_standalone.yaml"


def _download_videos(result: dict, local_save_path: str) -> None:
    local_dir = Path(local_save_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    downloaded = 0
    for ep in result.get("episodes") or []:
        if ep.get("video_path"):
            rel = volume_relative_path(ep["video_path"])
            local_file = local_dir / Path(rel).name
            try:
                data = b"".join(vol.read_file(rel))
                local_file.write_bytes(data)
                print(f"  downloaded: {local_file}")
                downloaded += 1
            except Exception:
                print(f"  skipped (not found): {rel}")
    print(f"Downloaded {downloaded} video(s) → {local_dir.resolve()}")


@app.local_entrypoint()
def main(
    run_readme: str = "external beta rollout",
    standalone_beta_checkpoint: str = (
        "/experiments/final_experiments/pusht/termination_study"
        "/stage4/obs_positive_negative_separate/epoch_0100.ckpt"
    ),
    high_checkpoint: str = HIGH_CHECKPOINT,
    low_checkpoint: str = LOW_CHECKPOINT,
    config: str = CONFIG_PATH,
    n_episodes: int = 5,
    test_start_seed: int = 100000,
    n_action_steps: int = 8,
    beta_transition: float = 0.92,
    max_steps: int = 300,
    video_failure_threshold: float = 20.0,
    local_save_path: str = "tmp/external_beta_rollout",
) -> None:
    result = spawn_modal_function(
        rollout_hierarchical_external_beta,
        label="rollout_external_beta",
        config_path=config,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
        standalone_beta_checkpoint=standalone_beta_checkpoint,
        n_episodes=n_episodes,
        test_start_seed=test_start_seed,
        n_action_steps=n_action_steps,
        beta_transition=beta_transition,
        max_steps=max_steps,
        no_video=False,
        video_failure_threshold=video_failure_threshold,
        invoke_command=run_readme,
    )

    print(f"\n--- external beta rollout ---")
    print(f"β ckpt: {standalone_beta_checkpoint}")
    print(f"n_episodes={result.get('n_episodes')}  mean_max_overlap={result.get('mean_max_overlap', 0):.1f}%")
    for ep in result.get("episodes") or []:
        m = ep.get("metrics") or {}
        print(
            f"  ep={ep['episode_idx']} seed={ep['seed']} "
            f"steps={ep['n_steps']} replans={ep['n_replans']} "
            f"max_overlap={m.get('max_overlap_full', 0):.1f}%"
        )

    _download_videos(result, local_save_path)
