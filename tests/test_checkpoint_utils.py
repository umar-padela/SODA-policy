"""Tests for resume-oriented checkpoint fields."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import pytest

import json

from soda.training.checkpoint_utils import (
    CHECKPOINT_FORMAT_VERSION,
    build_lr_schedule_meta,
    resume_training_payload,
    write_metrics_history,
)
from soda.training.train_low import save_checkpoint


class _FakeNormalizer:
    def state_dict(self):
        return {"scale": 1.0}


class _FakePolicy:
    horizon = 16

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(num_options=3)
        self.normalizer = _FakeNormalizer()

    def state_dict(self):
        return {"w": torch.tensor([1.0])}


def test_build_lr_schedule_meta():
    train_cfg = SimpleNamespace(
        num_epochs=100,
        lr_warmup_epochs=5,
        lr=1e-4,
        weight_decay=1e-6,
    )
    meta = build_lr_schedule_meta(train_cfg, warmup_epochs=5, cosine_epochs=95)
    assert meta["schedule_type"] == "linear_warmup_cosine_decay"
    assert meta["warmup_epochs"] == 5
    assert meta["cosine_epochs"] == 95
    assert meta["base_lr"] == 1e-4


def test_resume_training_payload_includes_scheduler_and_rng():
    scheduler = torch.optim.lr_scheduler.LinearLR(
        torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
        start_factor=1.0,
        total_iters=5,
    )
    ema = MagicMock()
    ema.state_dict.return_value = {"step": 3}

    payload = resume_training_payload(
        lr_scheduler=scheduler,
        lr_schedule={"warmup_epochs": 5},
        ema=ema,
        train_state={"best_val_score": 1.23},
        rng_state={"torch": torch.get_rng_state()},
    )
    assert payload["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert "lr_scheduler" in payload
    assert payload["lr_schedule"]["warmup_epochs"] == 5
    assert payload["ema"] == {"step": 3}
    assert payload["train_state"]["best_val_score"] == 1.23
    assert "rng_state" in payload


def test_write_metrics_history_atomic(tmp_path: Path):
    path = tmp_path / "metrics.json"
    write_metrics_history(path, [{"epoch": 1, "val_loss": 1.0}])
    assert json.loads(path.read_text(encoding="utf-8")) == [{"epoch": 1, "val_loss": 1.0}]
    write_metrics_history(
        path,
        [{"epoch": 1, "val_loss": 1.0}, {"epoch": 2, "val_loss": 0.5}],
    )
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 2


def test_save_checkpoint_writes_resume_fields(tmp_path: Path):
    from unittest.mock import patch

    policy = _FakePolicy()
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, total_iters=2
    )
    ema = MagicMock()
    ema.state_dict.return_value = {"step": 1}

    path = tmp_path / "test.ckpt"
    with patch("dataclasses.asdict", return_value={"num_options": 3}):
        save_checkpoint(
            path,
            policy,
            optimizer,
            epoch=7,
            cfg={"name": "soda_supervised"},
            lr_scheduler=scheduler,
            lr_schedule={"warmup_epochs": 5, "cosine_epochs": 95},
            ema=ema,
            train_state={"best_val_score": 0.5, "best_checkpoint_metric": "loss"},
            metrics={"val_loss": 0.5},
        )

    loaded = torch.load(path, map_location="cpu")
    assert loaded["epoch"] == 7
    assert loaded["checkpoint_format_version"] == CHECKPOINT_FORMAT_VERSION
    assert "lr_scheduler" in loaded
    assert loaded["lr_schedule"]["warmup_epochs"] == 5
    assert loaded["train_state"]["best_val_score"] == 0.5
    assert "rng_state" in loaded
    assert loaded["ema"] == {"step": 1}
