"""Tests for ``soda.dataset.option_start_dataset``."""

from __future__ import annotations

from pathlib import Path

import pytest

from soda.dataset.option_start_dataset import OptionStartDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
PUSHT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_option_start_dataset_anchors_at_segment_start():
    ds = OptionStartDataset(PUSHT_ZARR, n_obs_steps=2, val_ratio=0.0, seed=0)
    assert len(ds) > 0

    sample = ds[0]
    assert sample["obs"]["image"].shape == (2, 3, 96, 96)
    assert sample["obs"]["agent_pos"].shape == (2, 2)
    assert sample["option_id"].ndim == 0
    assert "action" not in sample
    assert "beta_label" not in sample

    seg = ds.segments[0]
    assert int(sample["anchor_index"]) == seg.start
    assert int(sample["segment_start"]) == seg.start
    assert int(sample["segment_end"]) == seg.end
    assert int(sample["option_id"]) == seg.option_id


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_option_start_validation_complement():
    ds = OptionStartDataset(PUSHT_ZARR, val_ratio=0.2, seed=0)
    val_ds = ds.get_validation_dataset()
    assert len(ds) + len(val_ds) > 0

    train_starts = {(s.start, s.end, s.option_id) for s in ds.segments}
    val_starts = {(s.start, s.end, s.option_id) for s in val_ds.segments}
    assert train_starts.isdisjoint(val_starts)


@pytest.mark.skipif(not PUSHT_ZARR.is_dir(), reason="pusht.zarr not present")
def test_option_start_multi_anchor_cut_last_n():
    """cut_last_n > 0 expands train samples; val is always single-anchor."""
    ds_single = OptionStartDataset(PUSHT_ZARR, val_ratio=0.1, seed=0, cut_last_n=0)
    ds_multi = OptionStartDataset(PUSHT_ZARR, val_ratio=0.1, seed=0, cut_last_n=8)

    # Same segments; more samples when cut_last_n > 0
    assert len(ds_multi.segments) == len(ds_single.segments)
    assert len(ds_multi) >= len(ds_single)

    # Val always uses single anchor regardless of cut_last_n
    val_single = ds_single.get_validation_dataset()
    val_multi = ds_multi.get_validation_dataset()
    assert len(val_multi) == len(val_multi.segments)
    assert len(val_multi) == len(val_single)

    # All anchors lie within their segment; option_id is consistent
    seen_non_start = False
    for seg, anchor, _ in ds_multi._samples:
        assert seg.start <= anchor < seg.end
        if anchor != seg.start:
            seen_non_start = True
    assert seen_non_start, "Expected at least one non-start anchor with cut_last_n=8"

    # Sample shape unchanged
    sample = ds_multi[0]
    assert sample["obs"]["image"].shape == (2, 3, 96, 96)
    assert sample["obs"]["agent_pos"].shape == (2, 2)
