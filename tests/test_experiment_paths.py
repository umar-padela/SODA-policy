"""Tests for task-first experiment directory layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from soda.eval.eval_manifest import build_eval_output_dir, run_dir_name_from_path
from soda.experiments.paths import (
    frozen_dp_pusht_checkpoint,
    infer_task_slug,
    local_mirror_for_experiments_path,
    segment_rollout_dir,
    sweep_low_dir,
    to_volume_experiments_path,
    train_low_dir,
    volume_relative_path,
    volume_metrics_remote_path,
)


def test_infer_task_slug_from_config_path():
    assert infer_task_slug(config_path=Path("configs/pusht/soda_supervised.yaml")) == "pusht"
    assert infer_task_slug(config_path=Path("configs/square/soda_supervised.yaml")) == "square"


def test_train_low_dir_local_and_volume():
    assert train_low_dir("pusht", "soda_supervised") == Path(
        "experiments/pusht/train_low/soda_supervised"
    )
    assert train_low_dir("pusht", "soda_supervised", in_volume=True) == (
        "/experiments/pusht/train_low/soda_supervised"
    )


def test_sweep_and_segment_paths():
    assert sweep_low_dir("pusht", "lr1.00e-04") == Path(
        "experiments/pusht/sweep_low/lr1.00e-04"
    )
    assert segment_rollout_dir("square") == Path("experiments/square/segment_rollout")


def test_to_volume_and_mirror():
    local = "experiments/pusht/sweep_low/lr1e-04"
    volume = "/experiments/pusht/sweep_low/lr1e-04"
    assert to_volume_experiments_path(local) == volume
    assert to_volume_experiments_path(volume) == volume
    repo = Path("/repo")
    assert local_mirror_for_experiments_path(volume, repo_root=repo) == (
        repo / "experiments/pusht/sweep_low/lr1e-04"
    )


def test_frozen_dp_pusht_checkpoint():
    assert frozen_dp_pusht_checkpoint(in_volume=True) == (
        "/experiments/pusht/dp_baselines/pusht_image_cnn_train0/latest.ckpt"
    )


def test_volume_relative_path():
    assert volume_relative_path("/experiments/pusht/eval/dp_frozen/20260524/031417") == (
        "pusht/eval/dp_frozen/20260524/031417"
    )


def test_volume_metrics_remote_path():
    assert volume_metrics_remote_path("/experiments/pusht/sweep_low/lr5.00e-04") == (
        "pusht/sweep_low/lr5.00e-04/metrics.json"
    )


def test_build_eval_output_dir_task_first():
    cfg = Path("configs/pusht/dp_frozen.yaml")
    out, _ = build_eval_output_dir(
        Path("/experiments"),
        config_path=cfg,
        timestamp=__import__("datetime").datetime(2026, 5, 24, 3, 14, 17),
    )
    assert out == Path("/experiments/pusht/eval/dp_frozen/20260524/031417")


def test_run_dir_name_from_path_task_first():
    out_dir = Path("/experiments/pusht/eval/dp_frozen/20260524/031417")
    assert run_dir_name_from_path(out_dir, Path("/experiments")) == (
        "pusht/dp_frozen/20260524/031417"
    )
