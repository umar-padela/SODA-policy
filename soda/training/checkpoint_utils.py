"""Shared training checkpoint fields for eval reload and future resume."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 1


def build_lr_schedule_meta(
    train_cfg: Any,
    *,
    warmup_epochs: int,
    cosine_epochs: int = 0,
) -> dict[str, Any]:
    """Serializable LR schedule metadata (rebuild or validate scheduler on resume)."""
    schedule_type = getattr(train_cfg, "lr_schedule_type", "cosine")
    beta_lr = getattr(train_cfg, "beta_lr", None)
    meta: dict[str, Any] = {
        "warmup_epochs": int(warmup_epochs),
        "base_lr": float(train_cfg.lr),
        "base_lr_beta": float(beta_lr) if beta_lr is not None else float(train_cfg.lr),
        "weight_decay": float(train_cfg.weight_decay),
    }
    if schedule_type == "plateau":
        meta["schedule_type"] = "linear_warmup_plateau"
        meta["plateau_patience"] = int(getattr(train_cfg, "lr_plateau_patience", 50))
        meta["plateau_factor"] = float(getattr(train_cfg, "lr_plateau_factor", 0.5))
        meta["plateau_min_lr"] = float(getattr(train_cfg, "lr_plateau_min_lr", 1e-6))
        meta["plateau_metric"] = str(getattr(train_cfg, "lr_plateau_metric", "loss"))
    else:
        meta["schedule_type"] = "linear_warmup_cosine_decay"
        meta["cosine_epochs"] = int(cosine_epochs)
        meta["num_epochs"] = int(train_cfg.num_epochs)
        meta["lr_warmup_epochs_config"] = int(train_cfg.lr_warmup_epochs)
    return meta


def capture_rng_state() -> dict[str, Any]:
    """Best-effort RNG snapshot for reproducible resume."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def resume_training_payload(
    *,
    lr_scheduler: Any | None = None,
    lr_scheduler_beta: Any | None = None,
    lr_schedule: dict[str, Any] | None = None,
    ema: Any | None = None,
    train_state: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extra checkpoint keys for training resume (not required for eval-only reload)."""
    payload: dict[str, Any] = {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
    }
    if lr_scheduler is not None:
        payload["lr_scheduler"] = lr_scheduler.state_dict()
    if lr_scheduler_beta is not None:
        payload["lr_scheduler_beta"] = lr_scheduler_beta.state_dict()
    if lr_schedule is not None:
        payload["lr_schedule"] = dict(lr_schedule)
    if ema is not None and callable(getattr(ema, "state_dict", None)):
        payload["ema"] = ema.state_dict()
    if train_state is not None:
        payload["train_state"] = dict(train_state)
    if rng_state is not None:
        payload["rng_state"] = rng_state
    return payload


def write_metrics_history(path: Path, history: list[dict[str, Any]]) -> None:
    """Persist per-epoch metrics (rewritten each epoch; cheap vs GPU train step)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    tmp.replace(path)
