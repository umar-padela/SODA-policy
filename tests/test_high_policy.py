"""Unit tests for π_high classifier (no DP checkpoint required)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from soda.models.high_policy import (
    HighPolicy,
    HighPolicyConfig,
    ObsEncoder,
    OptionClassifier,
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
        global_feat_dim=8,
        hidden_dim=32,
        num_layers=2,
    )
    defaults.update(overrides)
    return HighPolicyConfig(**defaults)


# ---------------------------------------------------------------------------
# OptionClassifier
# ---------------------------------------------------------------------------


def test_option_classifier_forward_shape() -> None:
    clf = OptionClassifier(global_feat_dim=16, hidden_dim=32, num_layers=2, num_options=3)
    feat = torch.randn(5, 16)
    logits = clf(feat)
    assert logits.shape == (5, 3)


def test_option_classifier_num_layers_one() -> None:
    # depth-1 is a single linear with no hidden block
    clf = OptionClassifier(global_feat_dim=8, hidden_dim=16, num_layers=1, num_options=4)
    logits = clf(torch.randn(2, 8))
    assert logits.shape == (2, 4)


def test_option_classifier_num_layers_raises() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        OptionClassifier(global_feat_dim=8, hidden_dim=16, num_layers=0, num_options=3)


# ---------------------------------------------------------------------------
# HighPolicy
# ---------------------------------------------------------------------------


def test_high_policy_loss_and_sample() -> None:
    obs_encoder = _dummy_obs_encoder()
    policy = HighPolicy(_cfg(), obs_encoder)
    B = 3
    obs = {
        "image": torch.rand(B, 2, 3, 96, 96),
        "agent_pos": torch.rand(B, 2, 2),
    }
    option_id = torch.tensor([0, 2, 1], dtype=torch.long)

    loss, logs = policy.compute_loss({"obs": obs, "option_id": option_id})
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logs["loss"] >= 0.0


def test_high_policy_forward_train_mode() -> None:
    # forward with option_id returns loss scalar
    obs_encoder = _dummy_obs_encoder()
    policy = HighPolicy(_cfg(), obs_encoder)
    obs = {"image": torch.rand(2, 2, 3, 96, 96), "agent_pos": torch.rand(2, 2, 2)}
    option_id = torch.tensor([0, 3], dtype=torch.long)
    loss = policy(obs, option_id)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_high_policy_sample_option_shape_and_range() -> None:
    obs_encoder = _dummy_obs_encoder()
    policy = HighPolicy(_cfg(num_options=3), obs_encoder)
    B = 5
    obs = {"image": torch.rand(B, 2, 3, 96, 96), "agent_pos": torch.rand(B, 2, 2)}
    global_feat = policy.encode_obs(obs)
    pred = policy.sample_option(global_feat)
    assert pred.shape == (B,)
    assert pred.dtype == torch.long
    assert pred.min() >= 0
    assert pred.max() < 3


def test_high_policy_forward_inference_mode() -> None:
    # forward without option_id returns option ids
    obs_encoder = _dummy_obs_encoder()
    policy = HighPolicy(_cfg(num_options=4), obs_encoder)
    B = 2
    obs = {"image": torch.rand(B, 2, 3, 96, 96), "agent_pos": torch.rand(B, 2, 2)}
    pred = policy(obs)
    assert pred.shape == (B,)
    assert pred.min() >= 0 and pred.max() < 4


def test_high_policy_sample_option_is_argmax() -> None:
    # sample_option must equal argmax of classifier logits
    obs_encoder = _dummy_obs_encoder()
    policy = HighPolicy(_cfg(num_options=3), obs_encoder)
    obs = {"image": torch.rand(2, 2, 3, 96, 96), "agent_pos": torch.rand(2, 2, 2)}
    global_feat = policy.encode_obs(obs)
    with torch.no_grad():
        logits = policy.classifier(global_feat)
        expected = logits.argmax(dim=-1)
        got = policy.sample_option(global_feat)
    assert torch.equal(got, expected)


def test_high_policy_loss_with_class_weights() -> None:
    obs_encoder = _dummy_obs_encoder()
    policy = HighPolicy(_cfg(num_options=3), obs_encoder)
    obs = {"image": torch.rand(4, 2, 3, 96, 96), "agent_pos": torch.rand(4, 2, 2)}
    option_id = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    weights = torch.tensor([1.0, 2.0, 3.0])
    loss, _ = policy.compute_loss({"obs": obs, "option_id": option_id}, class_weights=weights)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_high_policy_global_feat_dim_mismatch_raises() -> None:
    obs_encoder = _dummy_obs_encoder(feat_per_step=4)  # global_feat_dim=8
    with pytest.raises(ValueError, match="global_feat_dim"):
        HighPolicy(_cfg(global_feat_dim=99), obs_encoder)
