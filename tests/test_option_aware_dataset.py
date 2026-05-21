"""Tests for ``soda.dataset.option_aware_dataset``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soda.dataset.option_aware_dataset import (
    OptionLabeledZarrDataset,
    build_option_segment_index,
    derive_beta_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PUSHT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"


def test_build_option_segment_index_two_episodes():
    option_ids = np.array([1, 1, 2, 3, 3, 3], dtype=np.int32)
    episode_ends = np.array([3, 6], dtype=np.int64)
    segs = build_option_segment_index(option_ids, episode_ends)
    assert len(segs) == 3
    assert [(s.start, s.end, s.option_id) for s in segs] == [
        (0, 2, 1),
        (2, 3, 2),
        (3, 6, 3),
    ]


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_option_dataset_item_shapes():
    import zarr

    root = zarr.open(str(PUSHT_ZARR), mode="r")
    option_ids = np.asarray(root["data"]["option_id_supervised"][:])
    episode_ends = np.asarray(root["meta"]["episode_ends"][:])
    max_seg = 0
    for s in build_option_segment_index(option_ids, episode_ends):
        max_seg = max(max_seg, s.length)

    ds = OptionLabeledZarrDataset(
        PUSHT_ZARR,
        horizon_stretch_max=max_seg,
        n_obs_steps=2,
        val_ratio=0.0,
        seed=0,
    )
    assert len(ds) > 0

    sample = ds[0]
    assert sample["obs"]["image"].shape == (2, 3, 96, 96)
    assert sample["obs"]["agent_pos"].shape == (2, 2)
    assert sample["action"].shape == (max_seg, 3)  # D=2 + duration channel
    assert sample["option_id"].ndim == 0
    assert sample["beta_label"].ndim == 0
    assert int(sample["segment_length"]) >= 1

    bounds = ds.get_option_segment_bounds()
    assert len(bounds) == len(ds)
    assert bounds[0][2] == int(sample["option_id"])


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_horizon_stretch_max_auto_from_data():
    ds = OptionLabeledZarrDataset(PUSHT_ZARR, horizon_stretch_max=None, n_obs_steps=2)
    assert ds.horizon_stretch_max >= 1
    sample = ds[0]
    assert sample["action"].shape[0] == ds.horizon_stretch_max
