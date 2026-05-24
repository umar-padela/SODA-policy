"""Tests for expert-anchored segment rollouts (no GPU / sim required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soda.dataset.option_aware_dataset import OptionSegment, build_option_segment_index
from soda.eval.segment_rollout import compose_side_by_side_frames
from soda.eval.segment_zarr import (
    REPRESENTATIVE_MIN_SEGMENT_LENGTH,
    resolve_segment,
    select_representative_segments,
    ZarrSegmentStore,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUSHT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"


def _mock_store(segments: list[OptionSegment]) -> ZarrSegmentStore:
    return ZarrSegmentStore(
        zarr_path=Path("mock.zarr"),
        label_key="option_id_supervised",
        option_ids=np.zeros(1, dtype=np.int32),
        episode_ends=np.array([1], dtype=np.int64),
        state=None,  # type: ignore[arg-type]
        img=None,  # type: ignore[arg-type]
        segments=segments,
    )


def test_resolve_segment_by_index():
    segs = [
        OptionSegment(start=0, end=5, option_id=0, episode_idx=10),
        OptionSegment(start=5, end=12, option_id=1, episode_idx=10),
    ]
    store = _mock_store(segs)
    seg, idx = resolve_segment(store, segment_index=1)
    assert idx == 1
    assert seg.option_id == 1


def test_resolve_segment_by_episode_and_option():
    segs = [
        OptionSegment(start=0, end=3, option_id=0, episode_idx=7),
        OptionSegment(start=3, end=8, option_id=1, episode_idx=7),
        OptionSegment(start=8, end=10, option_id=0, episode_idx=7),
    ]
    store = _mock_store(segs)
    seg, idx = resolve_segment(
        store, episode_idx=7, option_id=0, segment_rank=1
    )
    assert idx == 2
    assert seg.start == 8


def test_compose_side_by_side_pads_shorter_policy_to_full_expert():
    expert = [np.zeros((96, 96, 3), dtype=np.uint8) for _ in range(30)]
    policy = [np.ones((96, 96, 3), dtype=np.uint8) * 255 for _ in range(4)]
    frames = compose_side_by_side_frames(expert, policy)
    assert len(frames) == 30
    assert frames[0].shape == (96, 192, 3)
    assert frames[29].shape == (96, 192, 3)


def test_compose_side_by_side_pads_shorter_expert():
    expert = [np.zeros((96, 96, 3), dtype=np.uint8) for _ in range(3)]
    policy = [np.ones((96, 96, 3), dtype=np.uint8) * 255]
    frames = compose_side_by_side_frames(expert, policy)
    assert len(frames) == 3
    assert frames[0].shape == (96, 192, 3)
    assert frames[2].shape == (96, 192, 3)


def test_actions_for_env_strips_soda_duration_and_batch():
    from soda.eval.pusht_rollout import _actions_for_env

    native = np.ones((1, 4, 2), dtype=np.float32)
    out = _actions_for_env(
        {"action": np.ones((1, 8, 3), dtype=np.float32), "action_unstretched": native},
        8,
    )
    assert out.shape == (4, 2)

    vec = _actions_for_env({"action": np.ones((2, 16, 2), dtype=np.float32)}, 8)
    assert vec.shape == (2, 8, 2)


def test_select_representative_segments_two_per_skill():
    segs = [
        OptionSegment(start=0, end=10, option_id=0, episode_idx=1),
        OptionSegment(start=10, end=30, option_id=0, episode_idx=2),
        OptionSegment(start=30, end=50, option_id=0, episode_idx=3),
        OptionSegment(start=50, end=60, option_id=1, episode_idx=1),
        OptionSegment(start=60, end=80, option_id=1, episode_idx=4),
        OptionSegment(start=80, end=100, option_id=1, episode_idx=5),
        OptionSegment(start=100, end=110, option_id=2, episode_idx=2),
        OptionSegment(start=110, end=130, option_id=2, episode_idx=6),
        OptionSegment(start=130, end=150, option_id=2, episode_idx=7),
    ]
    store = _mock_store(segs)
    picks = select_representative_segments(store, per_skill=2, min_length=8)
    assert len(picks) == 6
    by_skill: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for idx, seg in picks:
        by_skill[seg.option_id].append(idx)
    assert len(by_skill[0]) == 2
    assert len(by_skill[1]) == 2
    assert len(by_skill[2]) == 2
    assert len(set(by_skill[0])) == 2
    assert len(set(by_skill[1])) == 2
    assert len(set(by_skill[2])) == 2


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_zarr_pusht_env_state_dim():
    store = ZarrSegmentStore.open(PUSHT_ZARR)
    assert len(store.segments) > 0
    seg = store.segments[0]
    state = store.pusht_env_state(seg.start)
    assert state.shape == (5,)
    frame = store.expert_rgb_frame(seg.start)
    assert frame.ndim == 3 and frame.shape[-1] == 3


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_pusht_env_state_adds_pi_to_block_angle():
    store = ZarrSegmentStore.open(PUSHT_ZARR)
    seg = store.segments[0]
    raw = np.asarray(store.state[seg.start], dtype=np.float64).reshape(-1)
    sim = store.pusht_env_state(seg.start)
    assert sim.shape == (5,)
    assert np.allclose(sim[:4], raw[:4])
    expected = (raw[4] + np.pi) % (2 * np.pi)
    assert np.isclose(sim[4], expected)


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_select_representative_segments_on_real_zarr():
    store = ZarrSegmentStore.open(PUSHT_ZARR)
    picks = select_representative_segments(
        store, per_skill=2, min_length=REPRESENTATIVE_MIN_SEGMENT_LENGTH
    )
    assert len(picks) == 6
    counts = {0: 0, 1: 0, 2: 0}
    for _idx, seg in picks:
        counts[seg.option_id] += 1
        assert seg.length > 25
    assert counts == {0: 2, 1: 2, 2: 2}


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_build_option_segment_index_matches_store_count():
    store = ZarrSegmentStore.open(PUSHT_ZARR)
    rebuilt = build_option_segment_index(
        store.option_ids, store.episode_ends, min_segment_len=1
    )
    assert len(rebuilt) == len(store.segments)
