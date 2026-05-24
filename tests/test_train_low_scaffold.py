"""Tests for π_low training helpers (no full DP stack required)."""

from __future__ import annotations

import numpy as np

from soda.training.train_low import (
    TrainLowConfig,
    _aggregate_logs,
    _low_policy_config,
    _resolve_num_options,
    pusht_shape_meta,
)


class _FakeDataset:
    horizon = 16

    def __init__(self, option_ids: list[int]) -> None:
        self._option_ids = np.asarray(option_ids)


def test_train_low_config_from_hydra():
    cfg = {
        "train_low": {
            "num_epochs": 3,
            "batch_size": 8,
            "lr": 2e-4,
            "wandb_enabled": True,
        }
    }
    out = TrainLowConfig.from_hydra(cfg)
    assert out.num_epochs == 3
    assert out.batch_size == 8
    assert out.lr == 2e-4
    assert out.wandb_enabled is True


def test_pusht_shape_meta_action_dim():
    meta = pusht_shape_meta()
    assert meta["action"]["shape"] == [2]
    assert meta["obs"]["image"]["type"] == "rgb"


def test_low_policy_config_from_hydra():
    cfg = {
        "low_policy": {
            "num_options": 5,
            "option_embed_dim": 16,
            "bottleneck_dim": 1024,
            "termination_loss_weight": 0.5,
            "termination_input": "obs",
            "termination_stop_grad": False,
            "termination_head": {
                "bottleneck_dim": 1024,
                "hidden_dim": 128,
                "num_layers": 3,
            },
        }
    }
    ds = _FakeDataset([0, 1, 2, 1, 0])
    lp = _low_policy_config(cfg, ds, termination_pos_weight=10.0)
    assert lp.num_options == 5
    assert lp.termination_loss_weight == 0.5
    assert lp.termination_input == "obs"
    assert lp.termination_stop_grad is False
    assert lp.termination_pos_weight == 10.0
    assert lp.termination_head.num_layers == 3


def test_resolve_num_options_infers_from_dataset():
    cfg = {"low_policy": {"num_options": None}}
    ds = _FakeDataset([0, 1, 2, 1, 0])
    assert _resolve_num_options(cfg, ds) == 3


def test_aggregate_logs():
    out = _aggregate_logs(
        ["loss", "loss_diffusion"],
        {"loss": 6.0, "loss_diffusion": 4.0, "loss_termination": 2.0},
        3,
    )
    assert out["loss"] == 2.0
    assert out["loss_diffusion"] == 4.0 / 3.0
