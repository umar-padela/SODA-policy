"""Obs batching for Push-T eval matches DP ``BaseImagePolicy`` contract."""

import numpy as np

from soda.eval.pusht_rollout import _obs_to_torch


class _FakePolicy:
    device = "cpu"


def test_obs_to_torch_adds_batch_dim():
    n_obs = 2
    obs = {
        "image": np.zeros((n_obs, 3, 96, 96), dtype=np.float32),
        "agent_pos": np.zeros((n_obs, 2), dtype=np.float32),
    }
    out = _obs_to_torch(obs, _FakePolicy())
    assert out["image"].shape == (1, n_obs, 3, 96, 96)
    assert out["agent_pos"].shape == (1, n_obs, 2)
