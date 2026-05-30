"""Unit tests for hierarchical SODA controller (mock π_high / π_low)."""

from __future__ import annotations

import torch

from soda.inference.hierarchical_controller import (
    HierarchicalPolicy,
    HierarchicalPolicyConfig,
)


class _MockHigh:
    def __init__(self) -> None:
        self._next = 0

    def encode_obs(self, obs: dict) -> torch.Tensor:
        return torch.zeros(obs["image"].shape[0], 4)

    def sample_option(self, global_feat: torch.Tensor) -> torch.Tensor:
        b = int(global_feat.shape[0])
        opts = torch.arange(b) + self._next
        self._next += 1
        return opts


class _MockLow:
    def __init__(
        self,
        *,
        n_action_steps: int = 8,
        beta_after: int | None = None,
        native_steps: int = 16,
    ) -> None:
        self.n_action_steps = n_action_steps
        self.diffusion_calls = 0
        self.beta_calls = 0
        self._beta_after = beta_after
        self._step = 0
        self._native_steps = native_steps

    def predict_action(self, obs: dict, option_id: torch.Tensor) -> dict:
        self.diffusion_calls += 1
        b = int(option_id.shape[0])
        chunk = torch.zeros(b, 16, 3)
        for i in range(16):
            chunk[:, i, 0] = float(self.diffusion_calls * 100 + i)
            chunk[:, i, 1] = float(option_id.float().mean())
            chunk[:, i, 2] = 0.5
        native = torch.zeros(b, self._native_steps, 2)
        for i in range(self._native_steps):
            native[:, i, 0] = float(self.diffusion_calls * 100 + i)
            native[:, i, 1] = float(option_id.float().mean())
        cap = min(self._native_steps, self.n_action_steps)
        return {
            "action": native[:, :cap],
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
    ) -> torch.Tensor:
        self.beta_calls += 1
        self._step += 1
        if self._beta_after is not None and self._step == self._beta_after:
            return torch.ones(option_id.shape[0])
        return torch.zeros(option_id.shape[0])


def _obs(batch: int = 1) -> dict:
    return {
        "image": torch.zeros(batch, 2, 3, 96, 96),
        "agent_pos": torch.zeros(batch, 2, 2),
    }


def _policy(
    *,
    n_action_steps: int = 8,
    beta_after: int | None = None,
) -> HierarchicalPolicy:
    return HierarchicalPolicy(
        device=torch.device("cpu"),
        config=HierarchicalPolicyConfig(beta_transition=0.5, env_action_dim=2),
        pi_high=_MockHigh(),
        pi_low=_MockLow(n_action_steps=n_action_steps, beta_after=beta_after),
    )


def test_executes_full_native_before_replan():
    policy = HierarchicalPolicy(
        device=torch.device("cpu"),
        config=HierarchicalPolicyConfig(
            beta_transition=0.5,
            env_action_dim=2,
            n_action_steps=1,
        ),
        pi_high=_MockHigh(),
        pi_low=_MockLow(n_action_steps=16, beta_after=None, native_steps=4),
    )
    low = policy._pi_low
    assert isinstance(low, _MockLow)

    for _ in range(4):
        policy.predict_action(_obs())
    assert low.diffusion_calls == 1

    policy.predict_action(_obs())
    assert low.diffusion_calls == 2


def test_reuses_chunk_until_native_exhausted():
    policy = HierarchicalPolicy(
        device=torch.device("cpu"),
        config=HierarchicalPolicyConfig(
            beta_transition=0.5,
            env_action_dim=2,
            n_action_steps=8,
        ),
        pi_high=_MockHigh(),
        pi_low=_MockLow(n_action_steps=16, beta_after=None, native_steps=16),
    )
    low = policy._pi_low
    assert isinstance(low, _MockLow)

    first = policy.predict_action(_obs())
    assert low.diffusion_calls == 1
    assert first["action"].shape == (1, 1, 2)

    for _ in range(15):
        policy.predict_action(_obs())
    assert low.diffusion_calls == 1

    policy.predict_action(_obs())
    assert low.diffusion_calls == 2


def test_beta_checked_every_step_after_warmup():
    policy = HierarchicalPolicy(
        device=torch.device("cpu"),
        config=HierarchicalPolicyConfig(
            beta_transition=0.5,
            env_action_dim=2,
            n_action_steps=8,
        ),
        pi_high=_MockHigh(),
        pi_low=_MockLow(n_action_steps=16, beta_after=None),
    )
    low = policy._pi_low
    assert isinstance(low, _MockLow)

    policy.predict_action(_obs())
    assert low.beta_calls == 1

    for _ in range(5):
        policy.predict_action(_obs())
    assert low.beta_calls == 6


def test_beta_fires_early_replan():
    policy = HierarchicalPolicy(
        device=torch.device("cpu"),
        config=HierarchicalPolicyConfig(
            beta_transition=0.5,
            env_action_dim=2,
            n_action_steps=8,
        ),
        pi_high=_MockHigh(),
        pi_low=_MockLow(n_action_steps=16, beta_after=4),
    )
    low = policy._pi_low
    assert isinstance(low, _MockLow)

    for _ in range(3):
        policy.predict_action(_obs())
    assert low.diffusion_calls == 1

    policy.predict_action(_obs())
    assert low.diffusion_calls == 2


def test_replans_after_full_native_chunk():
    """Replan after native chunk exhausted, not at n_action_steps."""
    policy = HierarchicalPolicy(
        device=torch.device("cpu"),
        config=HierarchicalPolicyConfig(
            beta_transition=0.5,
            env_action_dim=2,
            n_action_steps=8,
        ),
        pi_high=_MockHigh(),
        pi_low=_MockLow(n_action_steps=16, beta_after=None, native_steps=16),
    )
    low = policy._pi_low
    assert isinstance(low, _MockLow)

    for _ in range(16):
        policy.predict_action(_obs())
    assert low.diffusion_calls == 1

    policy.predict_action(_obs())
    assert low.diffusion_calls == 2


def test_short_native_replans_after_exhausted():
    """N=4 native with n_action_steps=8: replan π_low after 4 steps."""
    policy = HierarchicalPolicy(
        device=torch.device("cpu"),
        config=HierarchicalPolicyConfig(
            beta_transition=0.5,
            env_action_dim=2,
            n_action_steps=8,
        ),
        pi_high=_MockHigh(),
        pi_low=_MockLow(n_action_steps=8, native_steps=4),
    )
    low = policy._pi_low
    assert isinstance(low, _MockLow)

    for _ in range(4):
        policy.predict_action(_obs())
    assert low.diffusion_calls == 1

    policy.predict_action(_obs())
    assert low.diffusion_calls == 2


def test_strips_duration_from_pre_sliced_action():
    policy = HierarchicalPolicy(
        device=torch.device("cpu"),
        config=HierarchicalPolicyConfig(
            beta_transition=0.5,
            env_action_dim=2,
            n_action_steps=1,
        ),
        pi_high=_MockHigh(),
        pi_low=_MockLow(n_action_steps=16, beta_after=None),
    )
    out = policy.predict_action(_obs())
    assert out["action"].shape == (1, 1, 2)
    assert out["action_pred"].shape == (1, 16, 3)
