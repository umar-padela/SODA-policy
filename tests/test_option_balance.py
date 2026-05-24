"""Tests for inverse-frequency option loss weighting."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soda.training.option_balance import (
    compute_inverse_freq_class_weights,
    compute_termination_pos_weight,
    resolve_option_class_weights,
    weighted_option_mean,
)


def test_inverse_freq_weights_equalize_class_contribution():
    option_ids = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    weights = compute_inverse_freq_class_weights(option_ids, num_options=3)

    assert weights.shape == (3,)
    # Normalized so count-weighted mean is 1.0: unnormalized N/(K*count) → [1, 2, 2/3]
    assert torch.allclose(
        torch.tensor([1.0, 2.0, 2.0 / 3.0], dtype=torch.float32), weights
    )

    per_sample = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ids = torch.tensor([0, 0, 1, 2, 2, 2])
    weighted = weighted_option_mean(per_sample, ids, weights)
    assert weighted.item() == pytest.approx(1.0)


def test_resolve_option_class_weights_none():
    ids = np.array([0, 1, 2])
    assert resolve_option_class_weights("none", ids, 3) is None
    assert resolve_option_class_weights(None, ids, 3) is None


def test_resolve_option_class_weights_unknown_mode():
    ids = np.array([0, 1])
    with pytest.raises(ValueError, match="Unknown option_balance"):
        resolve_option_class_weights("balanced_sampler", ids, 2)


def test_missing_class_raises():
    option_ids = np.array([0, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="zero train segments"):
        compute_inverse_freq_class_weights(option_ids, num_options=3)


def test_termination_pos_weight_from_frame_labels():
    option_ids = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    episode_ends = np.array([3, 5], dtype=np.int64)
    train_mask = np.array([True, True])
    gamma, n_neg, n_pos = compute_termination_pos_weight(
        option_ids, episode_ends, train_mask
    )
    assert n_pos == 2
    assert n_neg == 3
    assert gamma == pytest.approx(1.5)
