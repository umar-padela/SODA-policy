"""Tests for π_high ObsEncoder loading from π_low checkpoints (mocked, no full DP stack)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from soda.models.high_policy import ObsEncoder
from soda.training.train_high import TrainHighConfig, _resolve_low_checkpoint


class _FakeVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, nobs: dict[str, torch.Tensor]) -> torch.Tensor:
        n = int(nobs["image"].shape[0])
        return torch.zeros(n, 8, device=nobs["image"].device)


def _fake_low_policy(*, n_obs_steps: int = 2, feat_per_step: int = 8) -> SimpleNamespace:
    vision = _FakeVision()
    normalizer = MagicMock()
    normalizer.normalize.side_effect = lambda obs: obs
    return SimpleNamespace(
        obs_encoder=vision,
        normalizer=normalizer,
        n_obs_steps=n_obs_steps,
        obs_feature_dim=feat_per_step,
    )


@patch("soda.eval.policy_loaders.load_low_policy_from_checkpoint")
def test_obs_encoder_from_low_policy_checkpoint(mock_load: MagicMock) -> None:
    mock_load.return_value = _fake_low_policy(n_obs_steps=2, feat_per_step=8)

    enc = ObsEncoder.from_low_policy_checkpoint("/tmp/fake_low.ckpt", device="cpu")
    assert enc.source == "soda_low"
    assert enc.n_obs_steps == 2
    assert enc.global_feat_dim == 16
    assert not next(enc.vision_encoder.parameters()).requires_grad

    obs = {
        "image": torch.rand(2, 2, 3, 96, 96),
        "agent_pos": torch.rand(2, 2, 2),
    }
    out = enc(obs)
    assert out.shape == (2, 16)


@patch("soda.eval.policy_loaders.load_low_policy_from_checkpoint")
def test_load_frozen_low_obs_encoder_extracts_stack(mock_load: MagicMock) -> None:
    from soda.eval.policy_loaders import load_frozen_low_obs_encoder

    mock_load.return_value = _fake_low_policy(n_obs_steps=2, feat_per_step=8)
    vision, normalizer, n_obs, gfd = load_frozen_low_obs_encoder(
        "/tmp/x.ckpt", device="cpu"
    )
    assert vision is mock_load.return_value.obs_encoder
    assert normalizer is mock_load.return_value.normalizer
    assert n_obs == 2
    assert gfd == 16
    assert not next(vision.parameters()).requires_grad


def test_train_high_config_low_checkpoint_from_hydra():
    cfg = {"train_high": {"low_checkpoint": "experiments/train_low/soda_supervised/best.ckpt"}}
    out = TrainHighConfig.from_hydra(cfg)
    assert out.low_checkpoint == "experiments/train_low/soda_supervised/best.ckpt"


def test_resolve_low_checkpoint_requires_path(tmp_path):
    cfg = TrainHighConfig(low_checkpoint=None)
    with pytest.raises(FileNotFoundError, match="low_checkpoint is not set"):
        _resolve_low_checkpoint(cfg)

    missing = tmp_path / "missing.ckpt"
    cfg = TrainHighConfig(low_checkpoint=str(missing))
    with pytest.raises(FileNotFoundError, match="not found"):
        _resolve_low_checkpoint(cfg)
