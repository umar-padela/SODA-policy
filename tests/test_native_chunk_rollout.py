"""Tests for native-time π_low rollout adapters."""

from __future__ import annotations

import numpy as np
import torch

from soda.eval.low_only_runner import NativeChunkRolloutPolicy
from soda.eval.runner_common import extract_env_action_chunk


class _StubNativeLow:
    device = torch.device("cpu")
    dtype = torch.float32
    n_action_steps = 8
    calls = 0

    def reset(self) -> None:
        self.calls = 0

    def predict_action(self, obs_dict: dict) -> dict:
        batch = int(next(iter(obs_dict.values())).shape[0])
        tag = self.calls
        self.calls += 1
        native = torch.zeros(batch, 4, 2)
        for t in range(4):
            native[:, t, 0] = float(tag * 10 + t)
        stretched = torch.zeros(batch, 8, 3)
        return {
            "action": native[:, : self.n_action_steps],
            "action_pred": stretched,
            "action_unstretched": native,
        }


def test_extract_env_action_chunk_prefers_unstretched():
    native = np.arange(12, dtype=np.float32).reshape(1, 6, 2)
    out = extract_env_action_chunk(
        {"action": np.ones((1, 8, 3)), "action_unstretched": native},
        8,
        squeeze_batch=True,
    )
    assert out.shape == (6, 2)
    np.testing.assert_array_equal(out[0], native[0, 0])


def test_extract_env_action_chunk_caps_at_n_action_steps():
    native = np.arange(20, dtype=np.float32).reshape(1, 10, 2)
    out = extract_env_action_chunk(
        {"action_unstretched": native},
        8,
        squeeze_batch=True,
    )
    assert out.shape == (8, 2)


def test_native_chunk_rollout_reuses_diffusion_until_budget():
    inner = _StubNativeLow()
    policy = NativeChunkRolloutPolicy(inner)
    obs = {"image": torch.zeros(1, 2, 3, 96, 96)}

    first = policy.predict_action(obs)
    second = policy.predict_action(obs)
    third = policy.predict_action(obs)
    fourth = policy.predict_action(obs)

    assert first["action"].shape == (1, 1, 2)
    assert float(first["action"][0, 0, 0]) == 0.0
    assert float(second["action"][0, 0, 0]) == 1.0
    assert float(fourth["action"][0, 0, 0]) == 3.0
    assert inner.calls == 1

    fifth = policy.predict_action(obs)
    assert float(fifth["action"][0, 0, 0]) == 10.0
    assert inner.calls == 2


def test_low_policy_action_key_native_capped():
    """π_low ``action`` is native-time and capped at ``n_action_steps``."""
    inner = _StubNativeLow()
    obs = {"image": torch.zeros(1, 2, 3, 96, 96)}

    chunk = inner.predict_action(obs)
    assert inner.calls == 1
    assert chunk["action"].shape == (1, 4, 2)
    assert float(chunk["action"][0, -1, 0]) == 3.0
