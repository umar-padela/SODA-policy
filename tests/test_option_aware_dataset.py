"""Tests for ``soda.dataset.option_aware_dataset``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soda.dataset.option_aware_dataset import (
    OptionAwareDataset,
    OptionSegment,
    build_option_segment_index,
    derive_beta_labels,
    native_actions_from_anchor,
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


def test_native_actions_from_anchor_suffix():
    actions = np.arange(12, dtype=np.float32).reshape(6, 2)
    seg = OptionSegment(start=1, end=5, option_id=2, episode_idx=0)
    suffix = native_actions_from_anchor(actions, seg, anchor=3)
    assert suffix.shape == (2, 2)
    np.testing.assert_array_equal(suffix, actions[3:5])
    full = native_actions_from_anchor(actions, seg, anchor=1)
    assert full.shape == (4, 2)
    np.testing.assert_array_equal(full, actions[1:5])


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_option_dataset_suffix_action_at_fixed_anchor():
    import zarr

    root = zarr.open(str(PUSHT_ZARR), mode="r")
    option_ids = np.asarray(root["data"]["option_id_supervised"][:])
    episode_ends = np.asarray(root["meta"]["episode_ends"][:])
    max_seg = max(s.length for s in build_option_segment_index(option_ids, episode_ends))

    ds = OptionAwareDataset(
        PUSHT_ZARR,
        horizon=max_seg,
        n_obs_steps=2,
        val_ratio=0.0,
        seed=0,
        random_anchor=False,
    )
    seg = ds.segments[0]
    anchor = seg.end - 1
    sample = ds[0]
    assert int(sample["anchor_index"]) == anchor
    native = native_actions_from_anchor(root["data"]["action"], seg, anchor)
    expected = ds._stretcher.stretch_with_duration(native)
    np.testing.assert_allclose(sample["action"].numpy(), expected, rtol=1e-5, atol=1e-5)
    assert int(sample["segment_length"]) == 1
    assert float(sample["beta_label"]) == 1.0


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_train_val_episode_splits_are_disjoint():
    """Train and val must partition episodes (same seed, train vs ~mask)."""
    train_ds = OptionAwareDataset(
        PUSHT_ZARR,
        horizon=None,
        n_obs_steps=2,
        val_ratio=0.1,
        seed=42,
    )
    val_ds = train_ds.get_validation_dataset()

    train_eps = {seg.episode_idx for seg in train_ds.segments}
    val_eps = {seg.episode_idx for seg in val_ds.segments}
    assert train_eps.isdisjoint(val_eps), (
        f"overlap {len(train_eps & val_eps)} episodes between train and val"
    )
    assert train_eps | val_eps  # non-empty coverage


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_validation_dataset_inherits_random_anchor():
    train_ds = OptionAwareDataset(
        PUSHT_ZARR,
        horizon=None,
        n_obs_steps=2,
        val_ratio=0.1,
        seed=0,
        random_anchor=True,
    )
    val_ds = train_ds.get_validation_dataset()
    assert val_ds.random_anchor is True

    n = min(100, len(val_ds))
    betas = [float(val_ds[i]["beta_label"]) for i in range(n)]
    n_pos = sum(betas)
    assert 0 < n_pos < n, f"expected mixed beta labels on val, got {n_pos}/{n} positive"


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_option_dataset_item_shapes():
    import zarr

    root = zarr.open(str(PUSHT_ZARR), mode="r")
    option_ids = np.asarray(root["data"]["option_id_supervised"][:])
    episode_ends = np.asarray(root["meta"]["episode_ends"][:])
    max_seg = 0
    for s in build_option_segment_index(option_ids, episode_ends):
        max_seg = max(max_seg, s.length)

    ds = OptionAwareDataset(
        PUSHT_ZARR,
        horizon=max_seg,
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
def test_horizon_auto_from_data():
    ds = OptionAwareDataset(PUSHT_ZARR, horizon=None, n_obs_steps=2)
    assert ds.horizon >= 1
    sample = ds[0]
    assert sample["action"].shape[0] == ds.horizon


def _dp_stack_available() -> bool:
    if not (REPO_ROOT / "third_party" / "diffusion_policy" / "eval.py").is_file():
        return False
    try:
        import einops  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
@pytest.mark.skipif(not _dp_stack_available(), reason="diffusion_policy deps not available")
def test_get_normalizer_action_dim_matches_stretched_chunks():
    ds = OptionAwareDataset(PUSHT_ZARR, horizon=None, n_obs_steps=2)
    sample = ds[0]
    normalizer = ds.get_normalizer()
    normed = normalizer["action"].normalize(sample["action"].numpy())
    assert normed.shape == sample["action"].shape
