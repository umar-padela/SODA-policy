"""
Local entrypoint — expert-anchored π_low segment rollout on Modal.

Unlike ``modal_eval.py`` (random test seeds via PushTImageRunner), this resets sim
to a zarr expert segment start and writes a side-by-side expert | policy MP4.

Examples (repo root)::

  # List segments (CPU; no checkpoint needed)
  modal run modal/modal_rollout_low_policy.py --run-readme "list ep100 reposition" \\
    --list-segments --episode 100 --option-id 0

  # Auto-pick 2 median-length segments per skill (6 rollouts total)
  modal run modal/modal_rollout_low_policy.py --run-readme "qualitative panel" \\
    --representative-samples \\
    --config configs/pusht/soda_supervised.yaml

  # Rollout first REPOSITION segment in episode 100 (default checkpoint on Volume)
  modal run modal/modal_rollout_low_policy.py --run-readme "ep100 o0 segment rollout" \\
    --config configs/pusht/soda_supervised.yaml \\
    --episode 100 --option-id 0

  # Explicit checkpoint + segment index
  modal run modal/modal_rollout_low_policy.py --run-readme "seg 42 qual" \\
    --checkpoint /experiments/train_low/soda_supervised/epoch_0025.ckpt \\
    --segment-index 42

Outputs on Volume ``soda-experiments``::

  segment_rollout/seg*_ep*_o*_*.mp4
  segment_rollout/seg*_ep*_o*_*.json

Download::

  modal volume get soda-experiments segment_rollout/seg0001_ep100_o0_reposition_anchor12345.mp4 rollout.mp4
"""

from __future__ import annotations

from pathlib import Path

import modal

from modal_config import app, list_rollout_segments, rollout_low_policy, spawn_modal_function
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path


def _download_rollout_files(result: dict, local_save_path: str) -> None:
    """Download rollout output files from Modal Volume to a local directory."""
    local_dir = Path(local_save_path)
    local_dir.mkdir(parents=True, exist_ok=True)

    remote_paths: list[str] = []

    if result.get("output_dir"):
        rel = volume_relative_path(result["output_dir"])
        remote_paths.append(f"{rel}/invoke_command.txt")

    if result.get("representative_samples"):
        video_items = result.get("rollouts") or []
    else:
        video_items = [result] if result.get("video_path") else []

    for item in video_items:
        if item.get("video_path"):
            vrel = volume_relative_path(item["video_path"])
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
    checkpoint: str | None,
    list_segments: bool,
    representative_samples: bool,
    per_skill: int,
    segment_index: int | None,
    episode: int | None,
    option_id: int | None,
    segment_rank: int,
    fixed_option_id: int | None,
    label_key: str,
    n_action_steps: int,
    max_steps: int | None,
    output_dir: str | None,
    no_video: bool,
    list_limit: int,
    local_save_path: str | None,
) -> str:
    parts = [
        "modal run modal/modal_rollout_low_policy.py",
        f'--run-readme "{run_readme}"',
        f"--config {config}",
    ]
    if list_segments:
        parts.append("--list-segments")
    if representative_samples:
        parts.append("--representative-samples")
    if checkpoint:
        parts.append(f"--checkpoint {checkpoint}")
    if per_skill != 2:
        parts.append(f"--per-skill {per_skill}")
    if segment_index is not None:
        parts.append(f"--segment-index {segment_index}")
    if episode is not None:
        parts.append(f"--episode {episode}")
    if option_id is not None:
        parts.append(f"--option-id {option_id}")
    if segment_rank != 0:
        parts.append(f"--segment-rank {segment_rank}")
    if fixed_option_id is not None:
        parts.append(f"--fixed-option-id {fixed_option_id}")
    if label_key != "option_id_supervised":
        parts.append(f"--label-key {label_key}")
    if n_action_steps != 8:
        parts.append(f"--n-action-steps {n_action_steps}")
    if max_steps is not None:
        parts.append(f"--max-steps {max_steps}")
    if output_dir:
        parts.append(f"--output-dir {output_dir}")
    if no_video:
        parts.append("--no-video")
    if list_limit != 30:
        parts.append(f"--list-limit {list_limit}")
    if local_save_path:
        parts.append(f"--local-save-path {local_save_path}")
    return " ".join(parts)


@app.local_entrypoint()
def main(
    run_readme: str,
    config: str = "configs/pusht/soda_supervised.yaml",
    checkpoint: str | None = None,
    list_segments: bool = False,
    representative_samples: bool = False,
    per_skill: int = 2,
    segment_index: int | None = None,
    episode: int | None = None,
    option_id: int | None = None,
    segment_rank: int = 0,
    fixed_option_id: int | None = None,
    label_key: str = "option_id_supervised",
    n_action_steps: int = 8,
    max_steps: int | None = None,
    output_dir: str | None = None,
    no_video: bool = False,
    list_limit: int = 30,
    local_save_path: str | None = None,
):
    invoke_command = _build_invoke_command(
        run_readme=run_readme,
        config=config,
        checkpoint=checkpoint,
        list_segments=list_segments,
        representative_samples=representative_samples,
        per_skill=per_skill,
        segment_index=segment_index,
        episode=episode,
        option_id=option_id,
        segment_rank=segment_rank,
        fixed_option_id=fixed_option_id,
        label_key=label_key,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        output_dir=output_dir,
        no_video=no_video,
        list_limit=list_limit,
        local_save_path=local_save_path,
    )

    if list_segments:
        result = spawn_modal_function(
            list_rollout_segments,
            label="list_rollout_segments",
            episode_idx=episode,
            option_id=option_id,
            label_key=label_key,
            list_limit=list_limit,
        )
        print("\n--- list_rollout_segments ---")
        print(result.get("listing", ""))
        return

    if not representative_samples and segment_index is None and (
        episode is None or option_id is None
    ):
        raise SystemExit(
            "Pass --representative-samples, --segment-index, "
            "OR (--episode and --option-id). "
            "Use --list-segments to browse segments first."
        )

    result = spawn_modal_function(
        rollout_low_policy,
        label="rollout_low_policy",
        config_path=config,
        checkpoint=checkpoint,
        segment_index=segment_index,
        episode_idx=episode,
        option_id=option_id,
        segment_rank=segment_rank,
        fixed_option_id=fixed_option_id,
        label_key=label_key,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        output_dir=output_dir,
        no_video=no_video,
        representative_samples=representative_samples,
        per_skill=per_skill,
        invoke_command=invoke_command,
    )

    print("\n--- rollout_low_policy result ---")
    print(f"checkpoint: {result.get('checkpoint')}")
    print(f"output_dir: {result.get('output_dir')}")

    if result.get("representative_samples"):
        print(f"representative rollouts: {result.get('n_rollouts')} ({per_skill} per skill)")
        for item in result.get("rollouts") or []:
            metrics = item.get("metrics") or {}
            print(
                f"  seg={item.get('segment_index')} "
                f"skill={metrics.get('skill')} "
                f"ep={metrics.get('episode_idx')} "
                f"steps={item.get('n_policy_steps')} "
                f"max_overlap={metrics.get('max_overlap_full', 0):.1f}%"
            )
            if item.get("video_path"):
                rel = str(item["video_path"]).replace("/experiments/", "")
                print(f"    download: modal volume get soda-experiments {rel} tmp/{rel.split('/')[-1]}")
    else:
        print(
            f"segment_index: {result.get('segment_index')} "
            f"anchor: {result.get('anchor_frame')}"
        )
        metrics = result.get("metrics") or {}
        print(
            f"  steps={result.get('n_policy_steps')} "
            f"max_overlap={metrics.get('max_overlap_full', 0):.1f}% "
            f"skill={metrics.get('skill')}"
        )
        if result.get("video_path"):
            print(f"  video: {result['video_path']}")
            rel = str(result["video_path"]).replace("/experiments/", "")
            print(f"  download: modal volume get soda-experiments {rel} rollout.mp4")

    print("Done. Files on Modal Volume soda-experiments (under segment_rollout/).")

    if local_save_path:
        print(f"\nDownloading output files to {local_save_path} ...")
        _download_rollout_files(result, local_save_path)
