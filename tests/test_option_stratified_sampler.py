"""Tests for β stratified mini-batch sampling."""

from __future__ import annotations

from soda.dataset.option_stratified_sampler import (
    AllAnchorsBetaStratifiedBatchSampler,
    OptionBetaStratifiedBatchSampler,
    beta_batch_pos_fraction_from_gamma,
    decode_stratified_index,
    encode_stratified_index,
    positives_per_batch,
)


def test_beta_fraction_from_gamma():
    assert abs(beta_batch_pos_fraction_from_gamma(10.0) - 1.0 / 11.0) < 1e-9
    assert abs(beta_batch_pos_fraction_from_gamma(17.0) - 1.0 / 18.0) < 1e-9


def test_encode_decode_roundtrip():
    for seg_idx, at_end in ((0, False), (3, True), (99, False)):
        encoded = encode_stratified_index(seg_idx, at_segment_end=at_end)
        assert decode_stratified_index(encoded) == (seg_idx, at_end)


def test_stratified_batches_always_mix_beta_labels():
    batch_size = 64
    gamma = 10.0
    frac = beta_batch_pos_fraction_from_gamma(gamma)
    n_pos = positives_per_batch(batch_size, frac)
    sampler = OptionBetaStratifiedBatchSampler(
        num_segments=500,
        batch_size=batch_size,
        pos_fraction=frac,
        seed=0,
        drop_last=True,
    )
    for batch in sampler:
        n_beta_one = sum(1 for idx in batch if decode_stratified_index(idx)[1])
        n_beta_zero = len(batch) - n_beta_one
        assert n_beta_one == n_pos
        assert n_beta_zero == batch_size - n_pos
        assert n_beta_one >= 1
        assert n_beta_zero >= 1


def test_all_anchors_sampler_guarantees_pos_per_batch():
    # Simulate Push-T-like stats: 1433 pos, 24217 neg (5.6% pos rate)
    batch_size = 64
    pos_indices = list(range(1433))
    neg_indices = list(range(1433, 1433 + 24217))
    sampler = AllAnchorsBetaStratifiedBatchSampler(
        pos_indices=pos_indices,
        neg_indices=neg_indices,
        batch_size=batch_size,
        seed=0,
        drop_last=True,
    )
    expected_n_pos = sampler.pos_per_batch
    assert expected_n_pos >= 1
    assert expected_n_pos < batch_size
    for batch in sampler:
        assert len(batch) == batch_size
        n_pos = sum(1 for idx in batch if idx < 1433)
        assert n_pos == expected_n_pos, f"expected {expected_n_pos} pos, got {n_pos}"


def test_all_anchors_sampler_set_epoch_changes_order():
    pos_indices = list(range(50))
    neg_indices = list(range(50, 1000))
    sampler = AllAnchorsBetaStratifiedBatchSampler(
        pos_indices=pos_indices, neg_indices=neg_indices, batch_size=32, seed=42
    )
    sampler.set_epoch(0)
    batch_ep0 = list(sampler)[0]
    sampler.set_epoch(1)
    batch_ep1 = list(sampler)[0]
    assert batch_ep0 != batch_ep1, "different epochs should produce different batch order"
