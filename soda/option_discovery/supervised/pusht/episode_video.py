"""Render Push-T episodes with frame index, state, and skill overlays."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import zarr

from soda.option_discovery.supervised.pusht.frame_overlays import (
    SKILL_COLORS_BGR,
    burn_frame_number,
    burn_state_overlay,
)
from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES

PUSHT_FPS = 10

# Back-compat re-exports
__all__ = [
    "PUSHT_FPS",
    "SKILL_COLORS_BGR",
    "SKILL_NAMES",
    "burn_frame_number",
    "burn_state_overlay",
    "compile_frames_to_mp4",
    "compile_frames_to_grid",
    "episode_bounds",
    "extract_episode_frames_zarr",
    "open_video",
    "render_episode_video",
]


def episode_bounds(episode_ends: np.ndarray, episode_idx: int) -> tuple[int, int]:
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise IndexError(
            f"episode_idx {episode_idx} out of range [0, {len(episode_ends) - 1}]"
        )
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return start, end


def extract_episode_frames_zarr(
    root: zarr.Group,
    episode_idx: int,
    *,
    label_key: str = "option_id_supervised",
    fps: int = PUSHT_FPS,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    """Load one episode's RGB frames, state, and labels from an open zarr root."""
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    start, end = episode_bounds(episode_ends, episode_idx)

    img = root["data"]["img"]
    frames = [np.asarray(img[i], dtype=np.uint8).copy() for i in range(start, end)]
    state = np.asarray(root["data"]["state"][start:end], dtype=np.float32)
    labels = np.asarray(root["data"][label_key][start:end], dtype=np.int32)
    return frames, state, labels, fps


def compile_frames_to_mp4(
    frames: Sequence[np.ndarray],
    fps: int,
    output_path: str | Path,
    *,
    burn_frame_numbers: bool = False,
    states: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> Path:
    """Write RGB frames to an MP4 (OpenCV mp4v)."""
    if not frames:
        raise ValueError("frames is empty")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")

    for idx, frame in enumerate(frames):
        display = frame.copy()
        if burn_frame_numbers:
            display = burn_frame_number(display, idx)
        if states is not None and labels is not None:
            display = burn_state_overlay(display, states[idx], int(labels[idx]))
        bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        writer.write(bgr.astype(np.uint8))

    writer.release()
    return output_path


def compile_frames_to_grid(
    frames: Sequence[np.ndarray],
    output_path: str | Path = "video_grid.jpg",
) -> Path:
    """Tile frames into one JPEG grid (frame numbers burned in)."""
    if not frames:
        raise ValueError("frames is empty")

    output_path = Path(output_path)
    num_frames = len(frames)
    columns = math.ceil(math.sqrt(num_frames))
    rows = math.ceil(num_frames / columns)
    h, w = frames[0].shape[:2]

    grid = np.ones((rows * h, columns * w, 3), dtype=np.uint8) * 255
    for idx, frame in enumerate(frames):
        labeled = burn_frame_number(frame, idx)
        r, c = divmod(idx, columns)
        y, x = r * h, c * w
        grid[y : y + h, x : x + w] = labeled

    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return output_path


def open_video(path: Path) -> None:
    """Open MP4 with the OS default player."""
    path = path.resolve()
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def render_episode_video(
    zarr_path: Path,
    episode_idx: int,
    output_path: Path,
    *,
    label_key: str = "option_id_supervised",
    burn_frame_numbers: bool = True,
    burn_state: bool = True,
    open_player: bool = False,
) -> Path:
    """
    Export one episode to MP4 with overlays.

    Returns path to the written video.
    """
    root = zarr.open(str(zarr_path), mode="r")
    frames, states, labels, fps = extract_episode_frames_zarr(
        root, episode_idx, label_key=label_key
    )

    compile_frames_to_mp4(
        frames,
        fps,
        output_path,
        burn_frame_numbers=burn_frame_numbers,
        states=states if burn_state else None,
        labels=labels if burn_state else None,
    )

    n_episodes = len(root["meta"]["episode_ends"])
    print(
        f"Wrote {output_path} — episode {episode_idx}/{n_episodes - 1}, "
        f"{len(frames)} frames @ {fps} fps"
    )
    if open_player:
        open_video(output_path)
    return output_path
