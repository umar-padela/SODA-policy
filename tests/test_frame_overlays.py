"""Tests for Push-T frame overlay helpers."""

from __future__ import annotations

import numpy as np

from soda.option_discovery.supervised.pusht.frame_overlays import (
    LowPolicyOverlayInfo,
    apply_rollout_frame_overlays,
    format_beta_overlay,
    format_duration_frames_overlay,
    burn_low_policy_overlay,
    burn_option_overlay,
    duration_channel_stats,
)


def test_format_beta_overlay_shows_small_values():
    assert format_beta_overlay(0.0342, threshold=0.9) == "beta:0.0342 thr:0.90"


def test_format_duration_frames_overlay():
    assert format_duration_frames_overlay(0.42, 0.01, horizon=30) == "dur:12.6+-0.3"


def test_duration_channel_stats():
    horizon = 10
    chunk = np.zeros((horizon, 3), dtype=np.float32)
    chunk[:, -1] = 0.5
    mean, std, native = duration_channel_stats(chunk, horizon=horizon)
    assert mean == 0.5
    assert std == 0.0
    assert native == 5


def test_burn_low_policy_overlay_uses_ascii_labels():
    frame = np.full((96, 96, 3), 255, dtype=np.uint8)
    out = burn_low_policy_overlay(
        frame,
        LowPolicyOverlayInfo(
            beta=0.73,
            duration_mean=0.42,
            duration_std=0.01,
            horizon=30,
            decoded_native_steps=12,
            chunk_cursor=3,
            replan=True,
        ),
    )
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)


def test_burn_low_policy_overlay_writes_beta_and_duration():
    test_burn_low_policy_overlay_uses_ascii_labels()


def test_apply_rollout_frame_overlays_option_border():
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    out = apply_rollout_frame_overlays(
        frame,
        frame_index=7,
        option_id=1,
        low_policy_info=LowPolicyOverlayInfo(beta=0.2),
        burn_state=False,
    )
    bordered = burn_option_overlay(frame, 1)
    assert out.shape == frame.shape
    assert not np.array_equal(out, bordered)
