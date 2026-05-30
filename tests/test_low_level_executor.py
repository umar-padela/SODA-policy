"""Tests for shared π_low chunk executor."""

from __future__ import annotations

import torch

from soda.inference.low_level_executor import (
    HierarchicalPolicyConfig,
    LowLevelChunkExecutor,
)


class _MockLow:
    def __init__(self, *, native_steps: int = 4, beta_after: int | None = None) -> None:
        self.horizon = 16
        self.diffusion_calls = 0
        self.beta_calls = 0
        self._native_steps = native_steps
        self._beta_after = beta_after
        self._beta_step = 0

    def predict_action(self, obs: dict, option_id: torch.Tensor) -> dict:
        self.diffusion_calls += 1
        b = int(option_id.shape[0])
        chunk = torch.zeros(b, 16, 3)
        native = torch.zeros(b, self._native_steps, 2)
        for i in range(self._native_steps):
            native[:, i, 0] = float(self.diffusion_calls * 10 + i)
        return {
            "action": native,
            "action_pred": chunk,
            "action_unstretched": native,
        }

    def predict_beta(
        self,
        obs: dict,
        option_id: torch.Tensor,
        action_plan: torch.Tensor,
        *,
        diffusion_t: int = 0,
        chunk_cursor: float | None = None,
    ) -> torch.Tensor:
        self.beta_calls += 1
        self._beta_step += 1
        if self._beta_after is not None and self._beta_step >= self._beta_after:
            return torch.ones(option_id.shape[0])
        return torch.zeros(option_id.shape[0])


def _obs() -> dict:
    return {"image": torch.zeros(1, 2, 3, 96, 96)}


def test_executes_full_native_chunk_before_replan():
    low = _MockLow(native_steps=4)
    ex = LowLevelChunkExecutor(low, HierarchicalPolicyConfig(beta_transition=0.5))
    option = torch.zeros(1, dtype=torch.long)

    for _ in range(4):
        result = ex.step(_obs(), option)
        assert not result.beta_fired
        assert result.action.shape == (1, 1, 2)
    assert low.diffusion_calls == 1

    result = ex.step(_obs(), option)
    assert not result.beta_fired
    assert result.replanned
    assert low.diffusion_calls == 2


def test_rechecks_beta_after_replan_on_exhaustion():
    low = _MockLow(native_steps=1, beta_after=3)
    ex = LowLevelChunkExecutor(low, HierarchicalPolicyConfig(beta_transition=0.5))
    option = torch.zeros(1, dtype=torch.long)

    result = ex.step(_obs(), option)
    assert not result.beta_fired
    assert low.beta_calls == 1

    result = ex.step(_obs(), option)
    assert result.beta_fired
    assert low.beta_calls == 3


def test_beta_fires_without_executing_action():
    low = _MockLow(native_steps=8, beta_after=1)
    ex = LowLevelChunkExecutor(low, HierarchicalPolicyConfig(beta_transition=0.5))
    option = torch.zeros(1, dtype=torch.long)

    result = ex.step(_obs(), option)
    assert result.beta_fired
    assert result.action.numel() == 0
