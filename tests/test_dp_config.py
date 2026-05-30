"""Tests for dp.yaml → Columbia workspace config builder."""

from __future__ import annotations

import pytest

from soda.eval.policy_loaders import load_pusht_yaml
from soda.training.dp_config import (
    _dataset_train_n_action_steps,
    build_columbia_workspace_cfg,
    resolve_dp_output_dir,
)


def test_dataset_train_n_action_steps_from_pad_after():
    cfg = load_pusht_yaml("configs/pusht/dp.yaml")
    assert _dataset_train_n_action_steps(cfg) == 8


def test_build_columbia_workspace_cfg_uses_train_blocks():
    pytest.importorskip("omegaconf")
    cfg = load_pusht_yaml("configs/pusht/dp.yaml")
    ws = build_columbia_workspace_cfg(cfg)
    assert ws.horizon == 16
    assert ws.n_obs_steps == 2
    assert ws.n_action_steps == 8
    assert ws.training.num_epochs == 3050
    assert ws.policy.down_dims == [512, 1024, 2048]
    assert "DiffusionUnetHybridImagePolicy" in str(ws.policy._target_)
    assert ws.logging.mode == "online"
    assert ws.logging.project == "soda-train-dp"


def test_resolve_dp_output_dir_default():
    cfg = load_pusht_yaml("configs/pusht/dp.yaml")
    assert resolve_dp_output_dir(cfg).as_posix().endswith("experiments/pusht/train_dp/dp")
