"""Tests for ``soda.dataset.derive_beta_labels``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soda.dataset import derive_beta_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
PUSHT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"


def test_single_episode_multiple_segments():
    option_ids = np.array([1, 1, 1, 2, 2, 0, 0], dtype=np.int32)
    episode_ends = np.array([7], dtype=np.int32)
    beta = derive_beta_labels(option_ids, episode_ends)
    assert beta.dtype == np.float32
    assert beta.tolist() == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_two_episodes_segment_ends_do_not_cross_boundary():
    option_ids = np.array([1, 1, 1, 2, 2, 3, 3], dtype=np.int32)
    episode_ends = np.array([3, 7], dtype=np.int32)
    beta = derive_beta_labels(option_ids, episode_ends)
    assert beta.tolist() == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]


def test_invalid_episode_ends_raises():
    option_ids = np.zeros(5, dtype=np.int32)
    with pytest.raises(ValueError, match="episode_ends\\[-1\\]"):
        derive_beta_labels(option_ids, np.array([4], dtype=np.int32))


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_pusht_zarr_segment_count_matches_beta_ones():
    import zarr

    root = zarr.open(str(PUSHT_ZARR), mode="r")
    option_ids = np.asarray(root["data"]["option_id_supervised"][:])
    episode_ends = np.asarray(root["meta"]["episode_ends"][:])

    beta = derive_beta_labels(option_ids, episode_ends)

    n_segments = 0
    start = 0
    for ep_end in episode_ends:
        segment = option_ids[start:ep_end]
        if segment.size:
            n_segments += 1 + int(np.sum(segment[:-1] != segment[1:]))
        start = int(ep_end)

    assert int(beta.sum()) == n_segments
    assert beta.shape == option_ids.shape
