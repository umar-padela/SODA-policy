"""Tests for eval W&B helpers and yaml defaults."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from soda.eval.eval_yaml import build_eval_config_from_yaml
from soda.eval.policy_loaders import load_pusht_yaml
from soda.eval.wandb_logging import _numeric_metrics, log_eval_to_wandb
from soda.training.dp_config import build_columbia_workspace_cfg


def test_dp_frozen_yaml_wandb_defaults():
    cfg = build_eval_config_from_yaml("configs/pusht/dp_frozen.yaml")
    assert cfg.wandb_enabled is True
    assert cfg.wandb_project == "soda-eval"


def test_dp_yaml_train_wandb_defaults():
    pytest.importorskip("omegaconf")
    ws = build_columbia_workspace_cfg(load_pusht_yaml("configs/pusht/dp.yaml"))
    assert ws.logging.mode == "online"
    assert ws.logging.project == "soda-train-dp"


def test_numeric_metrics_skips_videos_and_bools():
    metrics = {
        "mean_score": 0.88,
        "n_episodes": 5,
        "video_paths": ["media/foo.mp4"],
        "ok": True,
    }
    assert _numeric_metrics(metrics) == {"mean_score": 0.88, "n_episodes": 5}


def test_log_eval_to_wandb_disabled_is_noop():
    log_eval_to_wandb({"metrics": {"mean_score": 1.0}}, enabled=False, project="soda-eval")


def test_log_eval_to_wandb_logs_metrics_and_videos(tmp_path: Path):
    video = tmp_path / "rollout.mp4"
    video.write_bytes(b"fake")

    fake_wandb = MagicMock()
    fake_wandb.Video = lambda path: f"Video({path})"

    with patch.dict("sys.modules", {"wandb": fake_wandb}):
        log_eval_to_wandb(
            {
                "descriptive_run_name": "pusht-dp-smoke",
                "config": {"n_test": 5},
                "metrics": {
                    "mean_score": 0.9,
                    "mean_score@125": 0.8,
                    "video_paths": [str(video)],
                },
            },
            enabled=True,
            project="soda-eval",
        )

    fake_wandb.init.assert_called_once()
    fake_wandb.log.assert_any_call({"mean_score": 0.9, "mean_score@125": 0.8})
    fake_wandb.finish.assert_called_once()
