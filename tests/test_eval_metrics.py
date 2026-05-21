"""Tests for Push-T eval metrics (no sim / GPU)."""

from __future__ import annotations

import numpy as np

from soda.eval.metrics import (
    DEFAULT_OVERLAP_CHECKPOINTS,
    aggregate_episode_metrics,
    compute_episode_metrics,
)


def test_prefix_max_overlap_at_checkpoints():
    # rewards are coverage in [0,1]; 50% at step 100, 90% at step 200
    rewards = [0.0] * 100 + [0.5] + [0.0] * 99 + [0.9] + [0.0] * 100
    ep = compute_episode_metrics(
        rewards,
        seed=42,
        checkpoints=(150, 200, 250, 300),
    )
    assert ep.max_overlap_at_step[150] == 50.0
    assert ep.max_overlap_at_step[200] == 50.0
    assert ep.max_overlap_at_step[250] == 90.0
    assert ep.max_overlap_at_step[300] == 90.0
    assert ep.max_overlap_full == 90.0


def test_aggregate_mean_score_scale():
    ep1 = compute_episode_metrics([0.8, 0.9], seed=0, checkpoints=(300,))
    ep2 = compute_episode_metrics([0.6, 1.0], seed=1, checkpoints=(300,))
    agg = aggregate_episode_metrics([ep1, ep2], checkpoints=DEFAULT_OVERLAP_CHECKPOINTS)
    assert agg["n_episodes"] == 2
    assert agg["mean_score"] == 95.0 / 100.0  # mean of 90, 100 pct -> 0.95
    assert agg["mean_score@300"] == 95.0 / 100.0
    assert "mean_max_overlap_pct" not in agg
    assert "success_rate" not in agg
