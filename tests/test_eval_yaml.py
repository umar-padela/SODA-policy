"""Tests for yaml-driven eval config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from soda.eval.eval_yaml import EvalCliOverrides, build_eval_config_from_yaml


def test_dp_frozen_yaml_smoke_defaults():
    cfg = build_eval_config_from_yaml("configs/pusht/dp_frozen.yaml")
    assert cfg.policy_source == "dp_frozen"
    assert cfg.config_name == "dp_frozen"
    assert cfg.n_test == 5
    assert cfg.max_steps == 300
    assert cfg.n_action_steps == 8
    assert str(cfg.checkpoint_path).endswith("latest.ckpt")


def test_dp_frozen_yaml_full_regime():
    cfg = build_eval_config_from_yaml(
        "configs/pusht/dp_frozen.yaml",
        cli=EvalCliOverrides(full=True),
    )
    assert cfg.n_test == 50


def test_cli_overrides_yaml():
    cfg = build_eval_config_from_yaml(
        "configs/pusht/dp_frozen.yaml",
        cli=EvalCliOverrides(n_test=1, n_action_steps=8),
    )
    assert cfg.n_test == 1
    assert cfg.n_action_steps == 8


def test_soda_supervised_yaml_policy_and_settings():
    cfg = build_eval_config_from_yaml(
        "configs/pusht/soda_supervised.yaml",
        finalize_checkpoints=False,
    )
    assert cfg.policy_source == "soda"
    assert cfg.config_name == "soda_supervised"
    assert cfg.n_action_steps == 8
    assert cfg.n_test == 5


def test_soda_yaml_checkpoint_paths(tmp_path: Path):
    high = tmp_path / "high.ckpt"
    low = tmp_path / "low.ckpt"
    high.write_bytes(b"x")
    low.write_bytes(b"y")

    cfg = build_eval_config_from_yaml(
        "configs/pusht/soda_supervised.yaml",
        cli=EvalCliOverrides(
            high_checkpoint=high,
            low_checkpoint=low,
        ),
    )
    assert cfg.high_checkpoint == high
    assert cfg.low_checkpoint == low
    assert cfg.checkpoint_path == high


def test_max_steps_from_env_runner():
    cfg = build_eval_config_from_yaml(
        "configs/pusht/soda_supervised.yaml",
        finalize_checkpoints=False,
    )
    assert cfg.max_steps == 300


def test_dp_yaml_infers_policy_dp():
    cfg = build_eval_config_from_yaml(
        "configs/pusht/dp.yaml",
        finalize_checkpoints=False,
    )
    assert cfg.policy_source == "dp"


def test_soda_yaml_missing_checkpoints_raises():
    with pytest.raises(ValueError, match="SODA eval requires"):
        build_eval_config_from_yaml("configs/pusht/soda_supervised.yaml")
