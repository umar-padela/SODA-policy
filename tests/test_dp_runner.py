"""Tests for DP runner action slicing (no GPU / sim)."""

from __future__ import annotations

import numpy as np

from soda.eval.dp_runner import slice_action_chunk


def test_slice_action_chunk_vectorized():
    action = np.arange(24, dtype=np.float32).reshape(2, 12, 1)
    out = slice_action_chunk(action, 1)
    assert out.shape == (2, 1, 1)
    np.testing.assert_array_equal(out[:, 0, 0], action[:, 0, 0])


def test_slice_action_chunk_eight_default():
    pred = np.zeros((1, 8, 2), dtype=np.float32)
    out = slice_action_chunk(pred, 8)
    assert out.shape == (1, 8, 2)
