"""Tests for ``soda.dataset.temporal_stretch.TemporalStretcher``."""

from __future__ import annotations

import numpy as np
import pytest

from soda.dataset.temporal_stretch import TemporalStretcher


@pytest.fixture
def stretcher() -> TemporalStretcher:
    return TemporalStretcher(horizon=16)


def test_stretch_shape_and_duration_normalized(stretcher: TemporalStretcher) -> None:
    actions = np.arange(21, dtype=np.float32).reshape(7, 3)
    stretched, d_norm = stretcher.stretch(actions)
    assert stretched.shape == (16, 3)
    assert d_norm == pytest.approx(7 / 16)


def test_stretch_segment_steps_mismatch_raises(stretcher: TemporalStretcher) -> None:
    actions = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        stretcher.stretch(actions, segment_steps=3)


def test_stretch_exceeds_horizon_raises(stretcher: TemporalStretcher) -> None:
    actions = np.zeros((20, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds horizon"):
        stretcher.stretch(actions)


def test_stretch_length_one(stretcher: TemporalStretcher) -> None:
    actions = np.array([[1.0, 2.0]], dtype=np.float32)
    stretched, d_norm = stretcher.stretch(actions)
    assert stretched.shape == (16, 2)
    assert d_norm == pytest.approx(1 / 16)
    np.testing.assert_allclose(stretched, np.tile(actions, (16, 1)), rtol=1e-5)


def test_stretch_identity_when_length_equals_horizon() -> None:
    s = TemporalStretcher(horizon=8)
    actions = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    stretched, d_norm = s.stretch(actions)
    np.testing.assert_allclose(stretched, actions, rtol=1e-5)
    assert d_norm == pytest.approx(1.0)


def test_round_trip_preserves_endpoints(stretcher: TemporalStretcher) -> None:
    actions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0]],
        dtype=np.float32,
    )
    stretched, _ = stretcher.stretch(actions, segment_steps=5)
    recovered = stretcher.unstretch(stretched, segment_steps_pred=5)
    assert recovered.shape == (5, 2)
    np.testing.assert_allclose(recovered[0], actions[0], rtol=0.05, atol=0.05)
    np.testing.assert_allclose(recovered[-1], actions[-1], rtol=0.05, atol=0.05)


def test_unstretch_clips_segment_steps_pred(stretcher: TemporalStretcher) -> None:
    actions = np.zeros((16, 2), dtype=np.float32)
    out = stretcher.unstretch(actions, segment_steps_pred=0)
    assert out.shape == (1, 2)
    out_hi = stretcher.unstretch(actions, segment_steps_pred=99)
    assert out_hi.shape == (16, 2)


def test_decode_duration_mean_pool(stretcher: TemporalStretcher) -> None:
    chunk = np.zeros((16, 3), dtype=np.float32)
    chunk[..., -1] = 7 / 16
    assert stretcher.decode_duration(chunk) == 7
    assert stretcher.decode_horizon(chunk) == 7


def test_decode_duration_clips(stretcher: TemporalStretcher) -> None:
    chunk = np.zeros((16, 2), dtype=np.float32)
    chunk[..., -1] = 0.01
    assert stretcher.decode_duration(chunk) == 1
    chunk[..., -1] = 1.5
    assert stretcher.decode_duration(chunk) == 16


def test_decode_segment_steps_batch() -> None:
    from soda.dataset.temporal_stretch import decode_segment_steps

    batch = np.zeros((2, 16, 3), dtype=np.float32)
    batch[0, :, -1] = 0.25
    batch[1, :, -1] = 1.0
    assert decode_segment_steps(batch, 16).tolist() == [4, 16]


def test_normalize_duration_alias() -> None:
    assert TemporalStretcher.normalize_horizon(4, 16) == 0.25
    assert TemporalStretcher.normalize_duration(4, 16) == 0.25


def test_stretch_batch(stretcher: TemporalStretcher) -> None:
    segs = [
        np.ones((3, 2), dtype=np.float32),
        np.ones((5, 2), dtype=np.float32) * 2,
    ]
    batch, durations = stretcher.stretch_batch(segs)
    assert batch.shape == (2, 16, 2)
    assert durations.tolist() == pytest.approx([3 / 16, 5 / 16])


def test_append_duration_channel(stretcher: TemporalStretcher) -> None:
    actions = np.zeros((16, 2), dtype=np.float32)
    out = stretcher.append_duration_channel(actions, 0.5)
    assert out.shape == (16, 3)
    assert np.all(out[:, -1] == pytest.approx(0.5))


def test_stretch_with_duration(stretcher: TemporalStretcher) -> None:
    actions = np.arange(6, dtype=np.float32).reshape(3, 2)
    out = stretcher.stretch_with_duration(actions)
    assert out.shape == (16, 3)
    assert out[0, -1] == pytest.approx(3 / 16)
