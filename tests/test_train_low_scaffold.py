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


def _minimal_train_low_block(**overrides) -> dict:
    """Minimal valid train_low config block (all required keys present)."""
    base = {
        "device": "cpu",
        "seed": 42,
        "batch_size": 8,
        "weight_decay": 1e-6,
        "checkpoint_every": 10,
        "output_dir": None,
        "wandb_enabled": False,
        "wandb_project": "soda-test",
        "wandb_run_name": None,
        "wandb_group": None,
        "wandb_tags": [],
        "finetune_dp_checkpoint": None,
        "num_workers": 2,
        "option_balance": "none",
        "use_ema": False,
        "lr_schedule_type": "plateau",
        "lr_plateau_patience": 10,
        "lr_plateau_patience_beta": None,
        "lr_plateau_factor": 0.5,
        "lr_plateau_min_lr": 1e-6,
        "beta_stratified_batches": False,
        "two_phase": False,
        "num_epochs": 3,
        "lr": 2e-4,
        "lr_warmup_epochs": 0,
        "beta_lr": None,
        "termination_loss_weight": 0.0,
        "termination_pos_weight": 1.0,
        "mask_diffusion_on_positive": False,
        "all_anchors": False,
        "best_checkpoint_metric": "loss_diffusion",
        "warmstart_unet": None,
        "checkpoint": None,
        "skip_diffusion_loss": False,
        "termination_stop_grad": True,
        "run_readme": None,
    }
    base.update(overrides)
    return base


def test_train_low_config_from_hydra():
    cfg = {"train_low": _minimal_train_low_block(num_epochs=3, batch_size=8, lr=2e-4, wandb_enabled=True)}
    out = TrainLowConfig.from_hydra(cfg)
    assert out.num_epochs == 3
    assert out.batch_size == 8
    assert out.lr == 2e-4
    assert out.wandb_enabled is True


def test_train_low_config_wandb_sweep_fields():
    cfg = {
        "train_low": _minimal_train_low_block(
            wandb_project="soda-sweep-low",
            wandb_run_name="lr3.00e-05",
            wandb_group="A4-lr-sweep",
            wandb_tags="sweep,pi_low,A4",
        )
    }
    out = TrainLowConfig.from_hydra(cfg)
    assert out.wandb_project == "soda-sweep-low"
    assert out.wandb_run_name == "lr3.00e-05"
    assert out.wandb_group == "A4-lr-sweep"
    assert out.wandb_tags == ("sweep", "pi_low", "A4")


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
    lp = _low_policy_config(cfg, ds, termination_loss_weight=0.5, termination_pos_weight=10.0)
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
