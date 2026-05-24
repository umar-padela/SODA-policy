"""Tests for hyperparameter sweep helpers."""

from __future__ import annotations

import json
from pathlib import Path

from soda.training.sweep_common import (
    default_low_lr_trials,
    format_lr_tag,
    local_mirror_for_experiments_path,
    modal_sweep_overrides,
    rank_high_trials,
    rank_low_trials,
    to_volume_experiments_path,
    trial_rank_paths,
)


def test_format_lr_tag():
    assert format_lr_tag(1e-4) == "lr1e-04"
    assert format_lr_tag(5e-5) == "lr5e-05"


def test_default_low_lr_trials_count():
    trials = default_low_lr_trials(num_epochs=5)
    assert len(trials) == 3
    assert trials[0].overrides["train_low.batch_size"] == 64
    assert trials[0].overrides["train_low.output_dir"] == "experiments/sweep_low/lr1e-04"


def test_to_volume_experiments_path():
    assert to_volume_experiments_path("experiments/sweep_low/lr1e-04") == (
        "/experiments/sweep_low/lr1e-04"
    )
    assert to_volume_experiments_path("/experiments/sweep_low/lr1e-04") == (
        "/experiments/sweep_low/lr1e-04"
    )
    assert to_volume_experiments_path(
        "experiments/train_low/soda_supervised/best.ckpt"
    ) == "/experiments/train_low/soda_supervised/best.ckpt"


def test_modal_sweep_overrides():
    overrides = {
        "train_high.output_dir": "experiments/sweep_high/lr1e-04",
        "train_high.low_checkpoint": "experiments/train_low/soda_supervised/best.ckpt",
        "train_high.lr": 1e-4,
    }
    out = modal_sweep_overrides(overrides)
    assert out["train_high.output_dir"] == "/experiments/sweep_high/lr1e-04"
    assert out["train_high.low_checkpoint"] == (
        "/experiments/train_low/soda_supervised/best.ckpt"
    )
    assert out["train_high.lr"] == 1e-4


def test_trial_rank_paths():
    trials = default_low_lr_trials(num_epochs=5)
    local_dir, volume_dir = trial_rank_paths(trials[0])
    assert volume_dir == "/experiments/sweep_low/lr1e-04"
    assert local_dir == local_mirror_for_experiments_path(volume_dir)
    assert local_dir.parts[-2:] == ("sweep_low", "lr1e-04")


def test_rank_low_trials(tmp_path: Path):
    dirs: list[tuple[str, Path]] = []
    for name, val_loss in (("a", 1.5), ("b", 1.2), ("c", 1.8)):
        out = tmp_path / name
        out.mkdir()
        history = [
            {
                "epoch": 5,
                "val_loss": val_loss,
                "val_loss_diffusion": val_loss - 1.0,
            }
        ]
        (out / "metrics.json").write_text(json.dumps(history), encoding="utf-8")
        dirs.append((name, out))

    ranked = rank_low_trials(dirs)
    assert [row["trial"] for row in ranked["val_loss"]] == ["b", "a", "c"]
    assert [row["trial"] for row in ranked["val_loss_diffusion"]] == ["b", "a", "c"]


def test_rank_low_trials_uses_best_epoch_not_final(tmp_path: Path):
    """Best total-loss epoch at 4 beats a trial whose final epoch dipped."""
    for name, history in (
        (
            "early_best",
            [
                {
                    "epoch": 4,
                    "val_loss": 1.5,
                    "val_loss_diffusion": 0.15,
                },
                {
                    "epoch": 5,
                    "val_loss": 2.0,
                    "val_loss_diffusion": 0.25,
                },
            ],
        ),
        (
            "final_dip",
            [
                {
                    "epoch": 4,
                    "val_loss": 1.95,
                    "val_loss_diffusion": 0.22,
                },
                {
                    "epoch": 5,
                    "val_loss": 1.9,
                    "val_loss_diffusion": 0.17,
                },
            ],
        ),
    ):
        out = tmp_path / name
        out.mkdir()
        (out / "metrics.json").write_text(json.dumps(history), encoding="utf-8")

    ranked = rank_low_trials(
        [("early_best", tmp_path / "early_best"), ("final_dip", tmp_path / "final_dip")]
    )
    by_total = ranked["val_loss"]
    assert by_total[0]["trial"] == "early_best"
    assert by_total[0]["best_epoch"] == 4
    assert by_total[0]["val_loss"] == 1.5
    assert by_total[1]["final_val_loss"] == 1.9


def test_rank_low_trials_can_disagree_on_winner(tmp_path: Path):
    """Total-loss and diffusion rankings may pick different trials."""
    histories = {
        "low_total": [
            {"epoch": 4, "val_loss": 1.0, "val_loss_diffusion": 0.50},
            {"epoch": 5, "val_loss": 2.0, "val_loss_diffusion": 0.10},
        ],
        "low_diffusion": [
            {"epoch": 4, "val_loss": 1.3, "val_loss_diffusion": 0.05},
            {"epoch": 5, "val_loss": 1.2, "val_loss_diffusion": 0.20},
        ],
    }
    dirs: list[tuple[str, Path]] = []
    for name, history in histories.items():
        out = tmp_path / name
        out.mkdir()
        (out / "metrics.json").write_text(json.dumps(history), encoding="utf-8")
        dirs.append((name, out))

    ranked = rank_low_trials(dirs)
    assert ranked["val_loss"][0]["trial"] == "low_total"
    assert ranked["val_loss_diffusion"][0]["trial"] == "low_diffusion"
    assert ranked["val_loss"][0]["best_epoch"] == 4
    assert ranked["val_loss_diffusion"][0]["best_epoch"] == 4


def test_rank_high_trials(tmp_path: Path):
    dirs: list[tuple[str, Path]] = []
    for name, acc in (("a", 0.4), ("b", 0.7), ("c", 0.55)):
        out = tmp_path / name
        out.mkdir()
        history = [{"epoch": 5, "val_option_acc": acc, "val_loss_fm": 1.0}]
        (out / "metrics.json").write_text(json.dumps(history), encoding="utf-8")
        dirs.append((name, out))

    ranked = rank_high_trials(dirs)
    assert [row["trial"] for row in ranked] == ["b", "c", "a"]
