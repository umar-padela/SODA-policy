"""
Local entrypoint — vanilla DP expert-anchored segment rollout on Modal.

Rolls out the Columbia frozen DP (no option conditioning, no β head) from the
same zarr-anchored segment start states used for π_low comparison. Useful for
checking whether diffusion quality is a ceiling imposed by the base DP.

Examples (repo root)::

  # 2 representative segments per skill (6 total), results auto-downloaded
  modal run modal/modal_rollout_dp_policy.py \\
    --run-readme "dp baseline segment comparison" \\
    --local-save-path experiments/pusht/segment_rollout/dp_baseline

  # Custom checkpoint or per-skill count
  modal run modal/modal_rollout_dp_policy.py \\
    --run-readme "dp alt checkpoint" \\
    --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt \\
    --per-skill 3 \\
    --local-save-path experiments/pusht/segment_rollout/dp_baseline

Outputs on Volume ``soda-experiments``::

  pusht/segment_rollout/dp_baseline/seg*_ep*_o*_*_dp.mp4
  pusht/segment_rollout/dp_baseline/seg*_ep*_o*_*_dp.json
"""

from __future__ import annotations

from pathlib import Path

import modal

from modal_config import app, rollout_dp_policy, spawn_modal_function
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path


def _download_rollout_files(result: dict, local_save_path: str) -> None:
    local_dir = Path(local_save_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    downloaded = 0
    for item in result.get("rollouts") or []:
        for suffix in (".mp4", ".json"):
            vpath = item.get("video_path", "")
            if not vpath:
                continue
            remote = volume_relative_path(vpath.replace(".mp4", suffix))
            local_file = local_dir / Path(remote).name
            try:
                data = b"".join(vol.read_file(remote))
                local_file.write_bytes(data)
                print(f"  downloaded: {local_file}")
                downloaded += 1
            except Exception:
                print(f"  skipped (not found): {remote}")
    print(f"Downloaded {downloaded} file(s) → {local_dir.resolve()}")


@app.local_entrypoint()
def main(
    run_readme: str,
    checkpoint: str | None = None,
    per_skill: int = 2,
    label_key: str = "option_id_supervised",
    n_action_steps: int = 8,
    max_steps: int | None = None,
    output_dir: str | None = None,
    no_video: bool = False,
    local_save_path: str | None = None,
):
    invoke_command = (
        f'modal run modal/modal_rollout_dp_policy.py --run-readme "{run_readme}"'
    )

    result = spawn_modal_function(
        rollout_dp_policy,
        label="rollout_dp_policy",
        checkpoint=checkpoint,
        label_key=label_key,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        output_dir=output_dir,
        no_video=no_video,
        per_skill=per_skill,
        invoke_command=invoke_command,
    )

    print("\n--- rollout_dp_policy result ---")
    print(f"checkpoint: {result.get('checkpoint')}")
    print(f"output_dir: {result.get('output_dir')}")
    print(f"rollouts: {result.get('n_rollouts')} ({per_skill} per skill)")
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
            rel = volume_relative_path(item["video_path"])
            print(f"    download: modal volume get soda-experiments {rel} tmp/{Path(rel).name}")

    if local_save_path:
        print(f"\nDownloading output files to {local_save_path} ...")
        _download_rollout_files(result, local_save_path)
