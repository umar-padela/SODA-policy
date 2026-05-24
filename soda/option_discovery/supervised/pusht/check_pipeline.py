#!/usr/bin/env python3
"""Smoke-test Push-T supervised tools against committed zarr (no download / GUI)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr

from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES, label_skills
from soda.option_discovery.supervised.pusht.build_zarr import _labeling_payload_from_state

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"
LABEL_KEY = "option_id_supervised"


def main() -> int:
    if not DEFAULT_ZARR.is_dir():
        print(f"FAIL: missing zarr at {DEFAULT_ZARR}", file=sys.stderr)
        return 1

    root = zarr.open(str(DEFAULT_ZARR), mode="r")
    labels = np.asarray(root["data"][LABEL_KEY][:], dtype=np.int32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int32)
    state = np.asarray(root["data"]["state"][:], dtype=np.float32)
    n_frames = int(labels.shape[0])
    n_eps = int(len(episode_ends))
    print(f"OK: zarr readable - {n_frames} frames, {n_eps} episodes, label key={LABEL_KEY}")

    unique = sorted(int(u) for u in np.unique(labels))
    expected = sorted(SKILL_NAMES.keys())
    if unique != expected:
        print(
            f"FAIL: expected skill ids {expected}, got {unique}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: skill ids {unique} ({[SKILL_NAMES[i] for i in unique]})")

    recomputed = label_skills(_labeling_payload_from_state(state, episode_ends))
    if not np.array_equal(recomputed, labels):
        print("FAIL: stored labels do not match label_skills(state)", file=sys.stderr)
        return 1
    print("OK: relabel consistency (state -> labels)")

    from soda.option_discovery.supervised.pusht.visualize_labels import visualize_labels
    from soda.option_discovery.supervised.pusht.episode_video import render_episode_video

    # Relabel consistency on a tiny synthetic episode
    fake = {
        "state": np.zeros((20, 5), dtype=np.float32),
        "episode_ends": np.array([20], dtype=np.int32),
    }
    out = label_skills(fake)
    assert out.shape == (20,)
    assert np.all(out == 0)
    print("OK: label_skills (synthetic)")

    with tempfile.TemporaryDirectory() as tmp:
        fig_path = Path(tmp) / "stats.png"
        visualize_labels(labels, episode_ends, show=False, save_path=fig_path)
        assert fig_path.is_file()
        print(f"OK: visualize_labels -> {fig_path.name}")

        vid_path = Path(tmp) / "ep0.mp4"
        render_episode_video(
            DEFAULT_ZARR,
            episode_idx=0,
            output_path=vid_path,
            label_key=LABEL_KEY,
            burn_frame_numbers=True,
            burn_state=True,
            open_player=False,
        )
        assert vid_path.is_file() and vid_path.stat().st_size > 1000
        print(f"OK: play_episode export -> {vid_path.name} ({vid_path.stat().st_size} bytes)")

    print("\nAll Push-T supervised pipeline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
