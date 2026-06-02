"""
Local entrypoint — full-episode hierarchical rollout (π_high + π_low + β) on Modal.

Produces annotated MP4s (frame index, skill border/label, β, duration channel,
chunk cursor, REPLAN indicator) and per-episode JSON summaries on the Volume,
then optionally downloads them to a local directory.

Examples (repo root)::

  # 5 episodes with annotated videos (defaults: n_action_steps=8, smoke seeds)
  modal run modal/modal_rollout_hierarchical.py --run-readme "hierarchical qual panel" \\
    --config configs/pusht/soda_supervised.yaml

  # Override n_action_steps for open-loop test and beta_transition to always replan
  modal run modal/modal_rollout_hierarchical.py --run-readme "open-loop test" \\
    --n-action-steps 100 --beta-transition 2.0 --n-episodes 3

  # Explicit checkpoints
  modal run modal/modal_rollout_hierarchical.py --run-readme "ep eval" \\
    --high-checkpoint /experiments/pusht/train_high/soda_supervised/best.ckpt \\
    --low-checkpoint  /experiments/pusht/train_low/soda_supervised/best.ckpt

  # Download videos to local dir
  modal run modal/modal_rollout_hierarchical.py --run-readme "download panel" \\
    --n-episodes 5 --local-save-path tmp/hierarchical_rollout

Outputs on Volume ``soda-experiments``::

  pusht/hierarchical_rollout/ep0000_seed100000.mp4
  pusht/hierarchical_rollout/ep0000_seed100000.json
  ...
"""

from __future__ import annotations

from pathlib import Path

import modal

from modal_config import app, rollout_hierarchical, spawn_modal_function
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path


def _download_rollout_files(result: dict, local_save_path: str) -> None:
    """Download hierarchical rollout MP4s and JSON summaries to a local directory."""
    local_dir = Path(local_save_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_paths: list[str] = []
    if result.get("output_dir"):
        rel = volume_relative_path(result["output_dir"])
        remote_paths.append(f"{rel}/invoke_command.txt")

    for ep in result.get("episodes") or []:
        if ep.get("video_path"):
            vrel = volume_relative_path(ep["video_path"])
            remote_paths.append(vrel)
            remote_paths.append(vrel.replace(".mp4", ".json"))

    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    downloaded = 0
    for remote in remote_paths:
        local_file = local_dir / Path(remote).name
        try:
            data = b"".join(vol.read_file(remote))
            local_file.write_bytes(data)
            print(f"  downloaded: {local_file}")
            downloaded += 1
        except Exception:
            print(f"  skipped (not found on volume): {remote}")

    print(f"Downloaded {downloaded} file(s) → {local_dir.resolve()}")


def _build_invoke_command(
    *,
    run_readme: str,
    config: str,
    high_checkpoint: str | None,
    low_checkpoint: str | None,
    n_episodes: int,
    test_start_seed: int,
    n_action_steps: int,
    beta_transition: float | None,
    open_loop: bool,
    duration_termination: bool,
    max_steps: int,
    output_dir: str | None,
    no_video: bool,
    video_failure_threshold: float | None,
    noise_eta: float,
    noise_rho: float,
    noise_scale_by_magnitude: bool,
    local_save_path: str | None,
) -> str:
    parts = [
        "modal run modal/modal_rollout_hierarchical.py",
        f'--run-readme "{run_readme}"',
        f"--config {config}",
    ]
    if high_checkpoint:
        parts.append(f"--high-checkpoint {high_checkpoint}")
    if low_checkpoint:
        parts.append(f"--low-checkpoint {low_checkpoint}")
    if n_episodes != 5:
        parts.append(f"--n-episodes {n_episodes}")
    if test_start_seed != 100000:
        parts.append(f"--test-start-seed {test_start_seed}")
    if n_action_steps != 8:
        parts.append(f"--n-action-steps {n_action_steps}")
    if beta_transition is not None:
        parts.append(f"--beta-transition {beta_transition}")
    if open_loop:
        parts.append("--open-loop")
    if duration_termination:
        parts.append("--duration-termination")
    if max_steps != 300:
        parts.append(f"--max-steps {max_steps}")
    if output_dir:
        parts.append(f"--output-dir {output_dir}")
    if no_video:
        parts.append("--no-video")
    if video_failure_threshold != 20.0:
        parts.append(f"--video-failure-threshold {video_failure_threshold}")
    if noise_eta != 0.0:
        parts.append(f"--noise-eta {noise_eta}")
    if noise_rho != 0.9:
        parts.append(f"--noise-rho {noise_rho}")
    if noise_scale_by_magnitude:
        parts.append("--noise-scale-by-magnitude")
    if local_save_path:
        parts.append(f"--local-save-path {local_save_path}")
    return " ".join(parts)


@app.local_entrypoint()
def main(
    run_readme: str,
    config: str = "configs/pusht/soda_supervised.yaml",
    high_checkpoint: str | None = None,
    low_checkpoint: str | None = None,
    n_episodes: int = 5,
    test_start_seed: int = 100000,
    n_action_steps: int = 8,
    beta_transition: float | None = None,
    open_loop: bool = False,
    duration_termination: bool = False,
    high_monitors_every_step: bool = False,
    max_steps: int = 300,
    output_dir: str | None = None,
    no_video: bool = False,
    video_failure_threshold: float | None = 20.0,
    noise_eta: float = 0.0,
    noise_rho: float = 0.9,
    noise_scale_by_magnitude: bool = False,
    local_save_path: str | None = None,
):
    """
    Run full-episode hierarchical rollout on Modal.

    Parameters
    ----------
    run_readme
        Short description archived in the output directory.
    config
        Path to SODA yaml (checkpoint paths + env settings).
    high_checkpoint, low_checkpoint
        Override yaml inference checkpoint paths.
    n_episodes
        Number of full episodes to roll out (default 5).
    test_start_seed
        First episode seed; ep i uses seed + i (default 100000, matching eval).
    n_action_steps
        Receding-horizon window (default 8). Ignored in open-loop mode.
    beta_transition
        Override beta_transition from yaml. Ignored in open-loop mode (set to 2.0 automatically).
    open_loop
        True: π_high resamples a new ω after every full native chunk completes.
        β is disabled. Each "segment" = one diffusion plan fully executed, then π_high picks next skill.
    max_steps
        Episode step budget (default 300, matching Columbia eval).
    video_failure_threshold
        Save video only for episodes scoring below this % (default 20.0). Set to None to save all.
    local_save_path
        If set, download MP4s and JSONs to this local directory after rollout.
    """
    invoke_command = _build_invoke_command(
        run_readme=run_readme,
        config=config,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
        n_episodes=n_episodes,
        test_start_seed=test_start_seed,
        n_action_steps=n_action_steps,
        beta_transition=beta_transition,
        open_loop=open_loop,
        duration_termination=duration_termination,
        max_steps=max_steps,
        output_dir=output_dir,
        no_video=no_video,
        video_failure_threshold=video_failure_threshold,
        noise_eta=noise_eta,
        noise_rho=noise_rho,
        noise_scale_by_magnitude=noise_scale_by_magnitude,
        local_save_path=local_save_path,
    )

    result = spawn_modal_function(
        rollout_hierarchical,
        label="rollout_hierarchical",
        config_path=config,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
        n_episodes=n_episodes,
        test_start_seed=test_start_seed,
        n_action_steps=n_action_steps,
        beta_transition=beta_transition,
        open_loop=open_loop,
        duration_termination=duration_termination,
        high_monitors_every_step=high_monitors_every_step,
        max_steps=max_steps,
        output_dir=output_dir,
        no_video=no_video,
        video_failure_threshold=video_failure_threshold,
        noise_eta=noise_eta,
        noise_rho=noise_rho,
        noise_scale_by_magnitude=noise_scale_by_magnitude,
        invoke_command=invoke_command,
    )

    print("\n--- rollout_hierarchical result ---")
    print(f"config: {result.get('config_path', config)}")
    print(f"π_high: {result.get('high_checkpoint')}")
    print(f"π_low:  {result.get('low_checkpoint')}")
    print(f"output: {result.get('output_dir')}")
    print(f"n_episodes: {result.get('n_episodes')} mean_max_overlap: {result.get('mean_max_overlap', 0):.1f}%")

    for ep in result.get("episodes") or []:
        m = ep.get("metrics") or {}
        print(
            f"  ep={ep['episode_idx']} seed={ep['seed']} "
            f"steps={ep['n_steps']} replans={ep['n_replans']} "
            f"max_overlap={m.get('max_overlap_full', 0):.1f}%"
        )
        if ep.get("video_path"):
            rel = str(ep["video_path"]).replace("/experiments/", "")
            print(
                f"    download: modal volume get soda-experiments {rel} "
                f"tmp/{rel.split('/')[-1]}"
            )

    print(
        "\nDone. Files on Modal Volume soda-experiments "
        "(under pusht/hierarchical_rollout/)."
    )

    if local_save_path:
        print(f"\nDownloading output files to {local_save_path} ...")
        _download_rollout_files(result, local_save_path)
