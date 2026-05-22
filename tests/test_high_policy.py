"""Unit tests for π_high (no DP checkpoint required)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from soda.models.high_policy import (
    HighPolicy,
    HighPolicyConfig,
    ObsEncoder,
    OptionFlowMatchingSchedule,
    OptionVelocityMLP,
)


class _DummyVision(nn.Module):
    """Returns fixed-size per-frame features for ObsEncoder smoke tests."""

    def __init__(self, feat_per_step: int = 4) -> None:
        super().__init__()
        self.feat_per_step = feat_per_step

    def forward(self, nobs: dict[str, torch.Tensor]) -> torch.Tensor:
        n = int(nobs["image"].shape[0])
        return torch.zeros(n, self.feat_per_step, device=nobs["image"].device)


class _DummyNormalizer:
    def normalize(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return obs


def _dummy_obs_encoder(*, n_obs_steps: int = 2, feat_per_step: int = 4) -> ObsEncoder:
    global_feat_dim = feat_per_step * n_obs_steps
    return ObsEncoder(
        _DummyVision(feat_per_step),
        _DummyNormalizer(),
        n_obs_steps,
        global_feat_dim=global_feat_dim,
        freeze=False,
    )


def _cfg(**overrides) -> HighPolicyConfig:
    defaults = dict(
        num_options=4,
        option_embed_dim=16,
        global_feat_dim=8,
        fm_hidden_dim=32,
        fm_num_layers=2,
        time_embed_dim=32,
        num_inference_steps=5,
    )
    defaults.update(overrides)
    return HighPolicyConfig(**defaults)


def test_high_policy_fm_loss_and_sample():
    obs_encoder = _dummy_obs_encoder()
    policy = HighPolicy(_cfg(), obs_encoder)
    B = 3
    obs = {
        "image": torch.rand(B, 2, 3, 96, 96),
        "agent_pos": torch.rand(B, 2, 2),
    }
    option_id = torch.tensor([0, 2, 1], dtype=torch.long)

    loss = policy(obs, option_id)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    pred = policy.sample_option(policy.encode_obs(obs))
    assert pred.shape == (B,)
    assert pred.dtype == torch.long
    assert pred.min() >= 0
    assert pred.max() < 4


def test_discrete_decode_nearest_embed():
    obs_encoder = _dummy_obs_encoder()
    policy = HighPolicy(_cfg(num_options=3, option_embed_dim=4), obs_encoder)
    # Place x_end exactly on option 1's embedding row.
    target = policy.option_embed.weight[1:2]
    pred = policy.discrete_decode_from_embedding(target)
    assert pred.item() == 1


@pytest.fixture
def velocity_net() -> OptionVelocityMLP:
    return OptionVelocityMLP(
        embed_dim=32,
        global_feat_dim=128,
        time_embed_dim=32,
        hidden_dim=256,
        num_layers=3,
    )


def test_option_velocity_mlp_forward_shape(velocity_net: OptionVelocityMLP) -> None:
    B, d, gf = 4, 32, 128
    x_t = torch.randn(B, d)
    global_feat = torch.randn(B, gf)
    t = torch.rand(B)
    v = velocity_net(x_t, global_feat, t)
    assert v.shape == (B, d)


def test_option_velocity_mlp_with_fm_interpolate(
    velocity_net: OptionVelocityMLP,
) -> None:
    B, d, gf = 4, 32, 128
    x1 = torch.randn(B, d)
    t = torch.rand(B)
    x_t, v_tgt = OptionFlowMatchingSchedule.interpolate(x1, t)
    assert x_t.shape == (B, d)
    assert v_tgt.shape == (B, d)
    v_pred = velocity_net(x_t, torch.randn(B, gf), t)
    assert v_pred.shape == (B, d)


def test_option_velocity_mlp_accepts_t_shape_b1(
    velocity_net: OptionVelocityMLP,
) -> None:
    B, d, gf = 2, 32, 128
    x_t = torch.randn(B, d)
    global_feat = torch.randn(B, gf)
    t = torch.rand(B, 1)
    v = velocity_net(x_t, global_feat, t)
    assert v.shape == (B, d)


def test_option_velocity_mlp_num_layers_raises() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        OptionVelocityMLP(32, 128, 32, hidden_dim=64, num_layers=0)


def test_option_flow_matching_interpolate_matches_hw1_formula() -> None:
    torch.manual_seed(0)
    x1 = torch.randn(16, 8)
    t = torch.rand(16)
    x_t, velocity = OptionFlowMatchingSchedule.interpolate(x1, t)
    x0 = x1 - velocity
    t_r = t.unsqueeze(-1)
    assert torch.allclose(x_t, t_r * x1 + (1 - t_r) * x0)
    assert torch.allclose(velocity, x1 - x0)


def test_option_flow_matching_integrate_euler_shape() -> None:
    B, d, gf = 3, 8, 16
    schedule = OptionFlowMatchingSchedule(embed_dim=d, num_inference_steps=4)
    global_feat = torch.randn(B, gf)

    def const_velocity(
        x_t: torch.Tensor, _gf: torch.Tensor, _t: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(x_t)

    x_end = schedule.integrate_euler(const_velocity, global_feat)
    assert x_end.shape == (B, d)
