"""Tests for eval run directory naming."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from soda.eval.run_naming import (
    build_eval_run_dir_name,
    checkpoint_slug,
    policy_label_from_config,
)


def test_policy_labels():
    assert policy_label_from_config("dp_baseline") == "diffusion_policy"
    assert policy_label_from_config("soda", "soda_supervised") == "soda_supervised"
    assert policy_label_from_config("soda", "soda_unsupervised") == "soda_unsupervised"


def test_frozen_dp_slug():
    path = Path("/experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt")
    assert checkpoint_slug(path) == "frozen_latest"


def test_train_run_slug():
    path = Path("/experiments/train_high/pusht_soda_supervised/epoch_0500.ckpt")
    slug = checkpoint_slug(path)
    assert "soda_supervised" in slug
    assert "epoch" in slug or "0500" in slug


def test_build_name_frozen_dp_smoke():
    ts = datetime(2026, 5, 22, 15, 30, 0, tzinfo=timezone.utc)
    name = build_eval_run_dir_name(
        task="pusht",
        policy_label="diffusion_policy",
        action_horizon=1,
        regime="receding",
        n_test=5,
        max_steps=300,
        checkpoint=Path("/experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt"),
        timestamp=ts,
    )
    assert name.startswith("pusht_diffusion_policy_h1_receding_smoke5_ckpt-frozen_latest_t300_utc20260522-153000")


def test_ckpt_slug_override():
    path = Path("/experiments/foo/bar.ckpt")
    assert checkpoint_slug(path, "my_run") == "my_run"
