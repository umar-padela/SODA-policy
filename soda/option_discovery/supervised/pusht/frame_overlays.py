"""Reusable Push-T frame overlays (frame index, skill, state, π_low debug)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES

# OpenCV uses BGR; matches histogram colors in visualize_labels (blue/green/orange)
SKILL_COLORS_BGR: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 165, 255),
}

OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_FONT_SCALE = 0.22
OVERLAY_THICKNESS = 1


@dataclass(frozen=True)
class LowPolicyOverlayInfo:
    """Debug fields for π_low rollouts (β + duration channel on current chunk)."""

    beta: float | None = None
    duration_mean: float | None = None  # normalized fraction in [0, 1] (mean over horizon rows)
    duration_std: float | None = None
    horizon: int | None = None  # U-Net sequence length; scales duration fraction → frames
    decoded_native_steps: int | None = None
    chunk_cursor: int | None = None
    replan: bool = False
    beta_threshold: float | None = None  # inference yaml; segment ends when beta >= this


def format_beta_overlay(beta: float, *, threshold: float | None = None) -> str:
    """Human-readable β with enough precision for small mid-segment values."""
    text = f"beta:{beta:.4f}"
    if threshold is not None:
        text += f" thr:{float(threshold):.2f}"
    return text


def format_duration_frames_overlay(
    duration_mean: float,
    duration_std: float | None,
    *,
    horizon: int,
) -> str:
    """Duration channel mean±std in native frames: ``dur:12.1+-0.2``."""
    mean_frames = float(duration_mean) * int(horizon)
    std_frames = 0.0 if duration_std is None else float(duration_std) * int(horizon)
    return f"dur:{mean_frames:.1f}+-{std_frames:.1f}"


LOW_POLICY_TEXT_BGR = (0, 0, 0)


def burn_frame_number(frame: np.ndarray, index: int) -> np.ndarray:
    """Frame index in top-left (red), matching ``episode_video``."""
    display = frame.copy()
    cv2.putText(
        display,
        str(index),
        (4, 8),
        OVERLAY_FONT,
        OVERLAY_FONT_SCALE,
        (255, 0, 0),
        OVERLAY_THICKNESS,
        cv2.LINE_AA,
    )
    return display


def burn_agent_block_state_overlay(frame: np.ndarray, state_vector: np.ndarray) -> np.ndarray:
    """Agent / block / rotation text (no skill border)."""
    display = frame.copy()
    ax, ay, bx, by, rot = np.round(np.asarray(state_vector, dtype=float).reshape(-1), 2)
    lines = [
        (f"A:{ax:.2f},{ay:.2f}", (0, 255, 0)),
        (f"B:{bx:.2f},{by:.2f}", (0, 255, 255)),
        (f"R:{rot:.2f}", (0, 0, 0)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            display,
            text,
            (16, 8 + i * 10),
            OVERLAY_FONT,
            OVERLAY_FONT_SCALE,
            color,
            OVERLAY_THICKNESS,
            cv2.LINE_AA,
        )
    return display


def burn_option_overlay(
    frame: np.ndarray,
    option_id: int,
    *,
    skill_names: Mapping[int, str] | None = None,
    skill_colors: Mapping[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Skill border + ``S:NAME`` label (same style as ``episode_video``)."""
    skill_names = skill_names or SKILL_NAMES
    skill_colors = skill_colors or SKILL_COLORS_BGR
    display = frame.copy()
    skill_color = skill_colors.get(int(option_id), (255, 255, 255))
    skill_name = skill_names.get(int(option_id), "UNKNOWN")
    cv2.rectangle(display, (0, 0), (95, 95), skill_color, 2)
    cv2.putText(
        display,
        f"S:{skill_name}",
        (4, 80),
        OVERLAY_FONT,
        OVERLAY_FONT_SCALE,
        skill_color,
        OVERLAY_THICKNESS,
        cv2.LINE_AA,
    )
    return display


def burn_state_overlay(
    frame: np.ndarray,
    state_vector: np.ndarray,
    option_id: int,
    skill_names: Mapping[int, str] | None = None,
    skill_colors: Mapping[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Agent/block/rotation text, skill border, and skill name on a 96x96 frame."""
    display = burn_agent_block_state_overlay(frame, state_vector)
    return burn_option_overlay(
        display,
        option_id,
        skill_names=skill_names,
        skill_colors=skill_colors,
    )


def burn_low_policy_overlay(
    frame: np.ndarray,
    info: LowPolicyOverlayInfo,
) -> np.ndarray:
    """β and duration-channel stats for the active diffusion chunk (ASCII for OpenCV)."""
    display = frame.copy()
    lines: list[str] = []
    if info.beta is not None:
        lines.append(
            format_beta_overlay(info.beta, threshold=info.beta_threshold)
        )
    if info.duration_mean is not None and info.horizon is not None:
        lines.append(
            format_duration_frames_overlay(
                info.duration_mean,
                info.duration_std,
                horizon=info.horizon,
            )
        )
    if info.decoded_native_steps is not None:
        lines.append(f"nat:{info.decoded_native_steps}")
    if info.chunk_cursor is not None:
        lines.append(f"cur:{info.chunk_cursor}")
    if info.replan:
        lines.append("REPLAN")

    # Middle-left of the 96x96 panel — avoids the side-by-side "policy" header at the top.
    y0 = 44
    for i, text in enumerate(lines):
        cv2.putText(
            display,
            text,
            (4, y0 + i * 11),
            OVERLAY_FONT,
            0.24,
            LOW_POLICY_TEXT_BGR,
            OVERLAY_THICKNESS,
            cv2.LINE_AA,
        )
    return display


def apply_rollout_frame_overlays(
    frame: np.ndarray,
    *,
    frame_index: int,
    option_id: int,
    state_vector: np.ndarray | None = None,
    low_policy_info: LowPolicyOverlayInfo | None = None,
    burn_state: bool = False,
) -> np.ndarray:
    """Compose standard segment-rollout overlays on one RGB frame."""
    display = burn_frame_number(frame, frame_index)
    if burn_state and state_vector is not None:
        display = burn_state_overlay(display, state_vector, option_id)
    else:
        display = burn_option_overlay(display, option_id)
    if low_policy_info is not None:
        display = burn_low_policy_overlay(display, low_policy_info)
    return display


def duration_channel_stats(
    action_pred: np.ndarray,
    *,
    horizon: int,
) -> tuple[float, float, int]:
    """
    Mean / std of the duration channel and decoded native segment length.

    ``action_pred`` is unnormalized ``(horizon, D+1)`` or ``(B, horizon, D+1)``.
    """
    from soda.dataset.temporal_stretch import decode_segment_steps

    chunk = np.asarray(action_pred, dtype=np.float64)
    if chunk.ndim == 3:
        chunk = chunk[0]
    if chunk.ndim != 2 or chunk.shape[-1] < 2:
        raise ValueError(f"expected (T, D+1) action_pred, got {chunk.shape}")
    duration = chunk[:, -1]
    native = int(decode_segment_steps(chunk, horizon))
    return float(np.mean(duration)), float(np.std(duration)), native
