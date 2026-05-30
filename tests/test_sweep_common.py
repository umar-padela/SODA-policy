"""Tests for hyperparameter sweep helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soda.training.sweep_common import (
    _training_subprocess_env,
    default_high_lr_trials,
    default_low_lr_grid,
    default_low_lr_trials,
    format_lr_tag,
    format_volume_metrics_get_command,
    local_mirror_for_experiments_path,
    modal_sweep_overrides,
    rank_high_trials,
    rank_low_trials,
    to_volume_experiments_path,
    trial_rank_paths,
)


def test_training_subprocess_env_sets_pythonpath():
    env = _training_subprocess_env(run_readme="test sweep")
    assert "PYTHONPATH" in env
    assert "soda" in env["PYTHONPATH"].lower() or "SODA-policy" in env["PYTHONPATH"]
    assert "diffusion_policy" in env["PYTHONPATH"]


def test_format_lr_tag():
    assert format_lr_tag(1e-4) == "lr1.00e-04"
    assert format_lr_tag(5e-5) == "lr5.00e-05"


def test_default_low_lr_grid():
    lrs = default_low_lr_grid()
    assert len(lrs) == 7
    assert lrs[0] == pytest.approx(3e-5, rel=1e-6)
    assert lrs[-1] == pytest.approx(5e-4, rel=1e-6)
    assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1))


def test_format_hydra_override_value():
    from soda.training.sweep_common import format_hydra_override_value, hydra_override_args

    assert format_hydra_override_value(("sweep", "pi_low", "A4")) == "[sweep,pi_low,A4]"
    assert format_hydra_override_value("sweep,pi_low,A4") == "'sweep,pi_low,A4'"
    assert hydra_override_args({"train_low.wandb_tags": ("sweep", "pi_low", "A4")}) == [
        "train_low.wandb_tags=[sweep,pi_low,A4]"
    ]


def test_default_low_lr_trials_count():
    trials = default_low_lr_trials(num_epochs=5)
    assert len(trials) == 7
    assert trials[0].overrides["train_low.batch_size"] == 64
    assert trials[0].overrides["train_low.lr"] == pytest.approx(3e-5, rel=1e-6)
    assert trials[0].overrides["train_low.wandb_project"] == "soda-sweep-low"
    assert trials[0].overrides["++train_low.wandb_run_name"] == "lr3.00e-05"
    assert trials[0].overrides["++train_low.wandb_group"] == "A4-lr-sweep"
    assert "sweep" in trials[0].overrides["++train_low.wandb_tags"]
    assert "lr3.00e-05" in str(trials[0].overrides["train_low.output_dir"])
    assert trials[-1].overrides["train_low.lr"] == pytest.approx(5e-4, rel=1e-6)
    assert trials[3].overrides["train_low.output_dir"] == "experiments/pusht/sweep_low/lr1.22e-04"


def test_default_high_lr_trials_wandb():
    trials = default_high_lr_trials(
        num_epochs=5,
        low_checkpoint="experiments/pusht/train_low/soda_supervised/best.ckpt",
    )
    assert len(trials) == 5
    assert trials[0].overrides["train_high.lr"] == pytest.approx(1e-5, rel=1e-6)
    assert trials[2].overrides["train_high.lr"] == pytest.approx(1e-4, rel=1e-6)
    assert trials[-1].overrides["train_high.lr"] == pytest.approx(1e-3, rel=1e-6)
    assert "high_policy.hidden_dim" not in trials[0].overrides
    assert trials[0].overrides["train_high.wandb_project"] == "soda-sweep-high"
    assert trials[0].overrides["++train_high.wandb_run_name"] == "lr1.00e-05"
    assert trials[0].overrides["++train_high.wandb_group"] == "B4-lr-sweep"


def test_to_volume_experiments_path():
    assert to_volume_experiments_path("experiments/pusht/sweep_low/lr1e-04") == (
        "/experiments/pusht/sweep_low/lr1e-04"
    )
    assert to_volume_experiments_path("/experiments/pusht/sweep_low/lr1e-04") == (
        "/experiments/pusht/sweep_low/lr1e-04"
    )
    assert to_volume_experiments_path(
        "experiments/pusht/train_low/soda_supervised/best.ckpt"
    ) == "/experiments/pusht/train_low/soda_supervised/best.ckpt"


def test_modal_sweep_overrides():
    overrides = {
        "train_high.output_dir": "experiments/pusht/sweep_high/lr1e-04",
        "train_high.low_checkpoint": "experiments/pusht/train_low/soda_supervised/best.ckpt",
        "train_high.lr": 1e-4,
    }
    out = modal_sweep_overrides(overrides)
    assert out["train_high.output_dir"] == "/experiments/pusht/sweep_high/lr1e-04"
    assert out["train_high.low_checkpoint"] == (
        "/experiments/pusht/train_low/soda_supervised/best.ckpt"
    )
    assert out["train_high.lr"] == 1e-4


def test_trial_rank_paths():
    trials = default_low_lr_trials(num_epochs=5)
    local_dir, volume_dir = trial_rank_paths(trials[3])
    assert volume_dir == "/experiments/pusht/sweep_low/lr1.22e-04"
    assert local_dir == local_mirror_for_experiments_path(volume_dir)
    assert local_dir.parts[-1] == "lr1.22e-04"


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


def test_format_volume_metrics_get_command():
    cmd = format_volume_metrics_get_command(
        volume_output_dir="/experiments/pusht/sweep_low/lr5.00e-04",
        local_output_dir=Path("experiments/pusht/sweep_low/lr5.00e-04"),
    )
    assert "modal volume get soda-experiments pusht/sweep_low/lr5.00e-04/metrics.json" in cmd
    assert "/experiments/" not in cmd.split("get soda-experiments ", 1)[1].split()[0]


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
