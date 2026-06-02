"""
Train standalone β network (ResNet + MLP) — termination_study stage 4.

Completely separate from train_low.py: no diffusion, no backbone/β optimizer split.
The k=9 LowPolicy checkpoint is used only to extract the obs normalizer; its weights
are not loaded. A fresh ResNet (ImageNet init) trains end-to-end via BCE loss.

Entrypoints
-----------
- Local: python soda/training/train_beta.py --config-path configs/pusht --config-name exp_term_standalone_resnet_positive_negative
- Modal: modal run experiments/final_experiments/pusht/termination_study/stage4_standalone_beta/modal_train_stage4.py
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainBetaConfig:
    # --- Infra ---
    device: str
    seed: int
    num_workers: int
    output_dir: str | None
    run_readme: str | None
    # --- Training ---
    num_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    checkpoint_every: int
    use_ema: bool
    best_checkpoint_metric: str         # "loss_termination" (lower=better)
    # --- Beta loss ---
    beta_pos_weight: float | None       # null → auto from dataset class frequencies
    label_smoothing: float              # 0.05 — matches other term experiments
    completion_weight: float | None     # null → disabled; ~2*(K-1)*avg_len to equalize natural β=1 vs escape
    # --- Data ---
    all_anchors: bool                   # True — required for escape_relabeling
    # --- W&B ---
    wandb_enabled: bool
    wandb_project: str
    wandb_run_name: str | None
    wandb_group: str | None
    wandb_tags: tuple[str, ...]
    # --- Source checkpoint (normalizer only) ---
    checkpoint: str | None             # k=9 ckpt — loads normalizer, NOT model weights

    @classmethod
    def from_hydra(cls, cfg: Any) -> TrainBetaConfig:
        block = _cfg_get(cfg, "train_beta", {})

        def _req(key: str) -> Any:
            v = _cfg_get(block, key, _MISSING)
            if v is _MISSING:
                raise ValueError(f"Required key 'train_beta.{key}' missing in config.")
            return v

        def _opt_float(key: str, default: Any = None) -> float | None:
            v = _cfg_get(block, key, default)
            return None if v is None else float(v)

        return cls(
            device=str(_req("device")),
            seed=int(_req("seed")),
            num_workers=int(_req("num_workers")),
            output_dir=_cfg_get(block, "output_dir", None),
            run_readme=_cfg_get(block, "run_readme", None),
            num_epochs=int(_req("num_epochs")),
            batch_size=int(_req("batch_size")),
            lr=float(_req("lr")),
            weight_decay=float(_req("weight_decay")),
            checkpoint_every=int(_req("checkpoint_every")),
            use_ema=bool(_req("use_ema")),
            best_checkpoint_metric=str(_req("best_checkpoint_metric")),
            beta_pos_weight=_opt_float("beta_pos_weight"),
            label_smoothing=float(_cfg_get(block, "label_smoothing", 0.05)),
            completion_weight=_opt_float("completion_weight"),
            all_anchors=bool(_cfg_get(block, "all_anchors", True)),
            wandb_enabled=bool(_req("wandb_enabled")),
            wandb_project=str(_req("wandb_project")),
            wandb_run_name=_cfg_get(block, "wandb_run_name", None),
            wandb_group=_cfg_get(block, "wandb_group", None),
            wandb_tags=_parse_wandb_tags(_cfg_get(block, "wandb_tags", None)),
            checkpoint=_cfg_get(block, "checkpoint", None),
        )


_MISSING = object()


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _parse_wandb_tags(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(p.strip() for p in raw.split(",") if p.strip())
    return tuple(str(p) for p in raw)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def build_beta_datasets(cfg: Any, *, all_anchors: bool = True):
    """Build OptionAwareDataset with escape_relabeling=True for standalone beta training."""
    from omegaconf import OmegaConf

    from soda.dataset.option_aware_dataset import build_option_dataset_from_config

    ds_raw = _cfg_get(_cfg_get(cfg, "task", {}), "dataset", {})
    try:
        ds_container = OmegaConf.to_container(ds_raw, resolve=True)
    except Exception:
        ds_container = dict(ds_raw) if ds_raw else {}
    if not isinstance(ds_container, dict):
        ds_container = {}

    ds_cfg = OmegaConf.create({
        "dataset": OmegaConf.merge(
            OmegaConf.create(ds_container),
            {
                "n_obs_steps": int(_cfg_get(cfg, "n_obs_steps", 2)),
                "horizon": _cfg_get(cfg, "horizon", None),
                "train": True,
                "random_anchor": True,
                "all_anchors": all_anchors,
                "escape_relabeling": True,  # always True for standalone beta
            },
        )
    })
    train_ds = build_option_dataset_from_config(ds_cfg)
    val_ds = train_ds.get_validation_dataset(all_anchors=all_anchors, escape_relabeling=True)
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def _build_standalone_beta_from_cfg(cfg: Any, train_ds: Any, *, normalizer: Any) -> Any:
    """Instantiate StandaloneBeta from Hydra config."""
    from omegaconf import OmegaConf

    from soda.models.standalone_beta import StandaloneBeta, StandaloneBetaConfig

    sb_raw = _cfg_get(cfg, "standalone_beta", {})
    try:
        sb_container = OmegaConf.to_container(sb_raw, resolve=True)
    except Exception:
        sb_container = dict(sb_raw) if sb_raw else {}

    num_options = int(train_ds.num_options)
    crop_raw = _cfg_get(sb_container, "crop_shape", [84, 84])

    sb_cfg = StandaloneBetaConfig(
        num_options=num_options,
        option_embed_dim=int(_cfg_get(sb_container, "option_embed_dim", 32)),
        n_obs_steps=int(_cfg_get(sb_container, "n_obs_steps", 2)),
        crop_shape=(int(crop_raw[0]), int(crop_raw[1])),
        obs_encoder_group_norm=bool(_cfg_get(sb_container, "obs_encoder_group_norm", True)),
        hidden_dim=int(_cfg_get(sb_container, "hidden_dim", 256)),
        num_layers=int(_cfg_get(sb_container, "num_layers", 2)),
        dropout_rate=float(_cfg_get(sb_container, "dropout_rate", 0.3)),
        use_chunk_cursor=bool(_cfg_get(sb_container, "use_chunk_cursor", False)),
    )

    return StandaloneBeta.build(sb_cfg, cfg, normalizer, imagenet_init=True)


def _load_normalizer_from_checkpoint(ckpt_path: str) -> Any:
    """Load only the obs normalizer from a LowPolicy checkpoint."""
    import dill
    from omegaconf import OmegaConf

    from soda.eval.policy_loaders import _ensure_diffusion_policy_path

    _ensure_diffusion_policy_path()

    payload = torch.load(ckpt_path, pickle_module=dill, map_location="cpu")
    # Handle combined checkpoints
    if "pi_low" in payload:
        payload = payload["pi_low"]

    if "normalizer" not in payload:
        raise ValueError(f"Checkpoint {ckpt_path} has no 'normalizer' key.")

    from diffusion_policy.model.common.normalizer import LinearNormalizer

    normalizer = LinearNormalizer()
    normalizer.load_state_dict(payload["normalizer"])
    print(f"[train_beta] Normalizer loaded from {ckpt_path}")
    return normalizer


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


def _group_grad_norm(module: Any) -> float:
    """L2 norm of all gradients in ``module`` (0.0 if frozen / no grads).

    A nonzero value for ``obs_encoder`` proves BCE gradients reach the ResNet —
    i.e. stop_grad is genuinely off and the backbone is being trained, not frozen.
    """
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return sq ** 0.5


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    pos_weight: float | None,
    label_smoothing: float,
    completion_weight: float | None = None,
    ema: Any | None = None,
) -> dict[str, float]:
    from soda.training.losses_low import termination_bce_loss

    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_norms: dict[str, float] = {}

    pw = torch.tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight is not None else None

    for batch in tqdm(loader, desc="train", leave=False):
        batch = _move_batch(batch, device)
        logit = model.forward_logit(batch["obs"], batch["option_id"])
        loss = termination_bce_loss(
            logit, batch["beta_label"],
            pos_weight=pw,
            label_smoothing=label_smoothing,
            is_escape=batch.get("is_escape"),
            completion_weight=completion_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Capture per-group grad norms on the first batch — confirms gradients reach the
        # ResNet obs_encoder (stop_grad=False) and not only the MLP head / option embedding.
        if n_batches == 0:
            grad_norms = {
                f"grad_norm/{name}": _group_grad_norm(getattr(model, name))
                for name in ("obs_encoder", "term_head", "option_embedding")
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.step(model)
        total_loss += float(loss.detach().item())
        n_batches += 1

    return {"loss_termination": total_loss / max(n_batches, 1), **grad_norms}


@torch.no_grad()
def validate(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    *,
    pos_weight: float | None,
    label_smoothing: float,
    completion_weight: float | None = None,
) -> dict[str, float]:
    from soda.training.losses_low import (
        BetaConfusionCounts,
        beta_confusion_counts_from_logits,
        beta_mae_from_logits,
        beta_metrics_from_counts,
        termination_bce_loss,
    )

    model.eval()
    total_loss = 0.0
    n_batches = 0
    beta_counts = BetaConfusionCounts()
    escape_counts = BetaConfusionCounts()
    natural_counts = BetaConfusionCounts()
    mae_sum = 0.0
    mae_n = 0

    pw = torch.tensor(pos_weight, dtype=torch.float32, device=device) if pos_weight is not None else None

    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch(batch, device)
        logit = model.forward_logit(batch["obs"], batch["option_id"])
        loss = termination_bce_loss(
            logit, batch["beta_label"],
            pos_weight=pw,
            label_smoothing=label_smoothing,
            is_escape=batch.get("is_escape"),
            completion_weight=completion_weight,
        )
        total_loss += float(loss.detach().item())
        n_batches += 1

        beta_counts += beta_confusion_counts_from_logits(logit, batch["beta_label"])
        s, n = beta_mae_from_logits(logit, batch["beta_label"])
        mae_sum += s
        mae_n += n

        is_escape = batch.get("is_escape")
        if is_escape is not None:
            esc_mask = is_escape.bool()
            nat_mask = ~esc_mask
            if esc_mask.any():
                escape_counts += beta_confusion_counts_from_logits(
                    logit[esc_mask], batch["beta_label"][esc_mask]
                )
            if nat_mask.any():
                natural_counts += beta_confusion_counts_from_logits(
                    logit[nat_mask], batch["beta_label"][nat_mask]
                )

    metrics = {"loss_termination": total_loss / max(n_batches, 1)}
    metrics.update(beta_metrics_from_counts(beta_counts))
    metrics["beta_mae"] = mae_sum / mae_n if mae_n > 0 else float("nan")
    for prefix, counts in (("escape", escape_counts), ("natural", natural_counts)):
        for k, v in beta_metrics_from_counts(counts).items():
            metrics[f"{prefix}_{k}"] = v
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_beta_checkpoint(
    path: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Any,
    *,
    ema_model: Any | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg_dict = dict(cfg) if isinstance(cfg, dict) else {}

    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict() if ema_model is not None else None,
        "optimizer": optimizer.state_dict(),
        "standalone_beta_config": asdict(model.cfg),
        "obs_feature_dim": model.obs_feature_dim,  # actual per-step encoder output dim
        "hydra_cfg": cfg_dict,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run_training(cfg: Any) -> None:
    from omegaconf import OmegaConf

    train_cfg = TrainBetaConfig.from_hydra(cfg)

    # Seed
    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train_beta] device={device}  epochs={train_cfg.num_epochs}  batch={train_cfg.batch_size}")

    # Output dir
    if train_cfg.output_dir:
        out_dir = Path(train_cfg.output_dir)
    else:
        out_dir = Path("experiments/train_beta") / str(_cfg_get(cfg, "name", "standalone_beta"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_beta] output_dir={out_dir}")

    # Normalizer from k=9 checkpoint
    if train_cfg.checkpoint is None:
        raise ValueError("train_beta.checkpoint must be set to the k=9 LowPolicy checkpoint path.")
    normalizer = _load_normalizer_from_checkpoint(train_cfg.checkpoint)

    # Datasets
    from soda.dataset.option_aware_dataset import OptionAwareDataset
    train_ds, val_ds = build_beta_datasets(cfg, all_anchors=train_cfg.all_anchors)
    print(f"[train_beta] train={len(train_ds)} / val={len(val_ds)} anchors")

    pin = str(device).startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=train_cfg.num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    # Model
    from soda.training.termination_utils import compute_auto_pos_weight

    model = _build_standalone_beta_from_cfg(cfg, train_ds, normalizer=normalizer)
    model.to(device)

    # pos_weight / completion_weight
    completion_weight = train_cfg.completion_weight
    pos_weight = train_cfg.beta_pos_weight
    num_options = int(train_ds.num_options)
    avg_len = sum(s.length for s in train_ds.segments) / len(train_ds.segments)

    if completion_weight is not None:
        print(f"[train_beta] completion_weight={completion_weight:.1f}  (natural β=1 upweight)  "
              f"K={num_options}, avg_len={avg_len:.1f}")

    if pos_weight is None:
        # Auto-calculate for escape_relabeling distribution.
        K = num_options
        L = avg_len
        pos_frac = (1 + (K - 1) * L) / (K * L)
        neg_frac = 1.0 - pos_frac
        pos_weight = round(neg_frac / pos_frac, 2)
        print(f"[train_beta] auto pos_weight={pos_weight:.3f}  (K={K}, avg_len={L:.1f})")
    else:
        print(f"[train_beta] pos_weight={pos_weight} (from config)")

    # Optimizer — single Adam over all params (obs_encoder + embedding + MLP all train together)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # EMA
    # Columbia DP's EMAModel takes the *shadow* model as `model=`: it calls
    # .eval()/.requires_grad_(False) on it and updates it in place via step(live_model).
    # Passing the live training model here would freeze its params, so loss.backward()
    # fails with "element 0 of tensors does not require grad and does not have a grad_fn".
    ema = ema_model = None
    if train_cfg.use_ema:
        from diffusion_policy.model.diffusion.ema_model import EMAModel
        ema_model = copy.deepcopy(model)
        ema = EMAModel(model=ema_model)

    # W&B
    wandb_run = None
    if train_cfg.wandb_enabled:
        try:
            import wandb
            from omegaconf import OmegaConf
            init_kwargs: dict[str, Any] = {
                "project": train_cfg.wandb_project,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "dir": str(out_dir),
            }
            if train_cfg.wandb_run_name:
                init_kwargs["name"] = train_cfg.wandb_run_name
            if train_cfg.wandb_group:
                init_kwargs["group"] = train_cfg.wandb_group
            if train_cfg.wandb_tags:
                init_kwargs["tags"] = list(train_cfg.wandb_tags)
            wandb_run = wandb.init(**init_kwargs)
        except Exception as e:
            print(f"[train_beta] W&B init failed: {e}")

    best_val_score: float | None = None
    metrics_log: list[dict] = []
    val_metrics: dict[str, float] = {}

    for epoch in range(1, train_cfg.num_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            pos_weight=pos_weight,
            label_smoothing=train_cfg.label_smoothing,
            completion_weight=completion_weight,
            ema=ema,
        )
        # ema.step(model) (called per batch in train_one_epoch) already updated ema_model
        # in place — it *is* ema.averaged_model — so validation below uses EMA weights.
        val_metrics = validate(
            ema_model if ema_model is not None else model,
            val_loader, device,
            pos_weight=pos_weight,
            label_smoothing=train_cfg.label_smoothing,
            completion_weight=completion_weight,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        combined = {"epoch": epoch, "train/lr": current_lr,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()}}
        metrics_log.append(combined)

        if wandb_run is not None:
            try:
                import wandb
                wandb.log(combined)
            except Exception:
                pass

        val_score = float(val_metrics.get(train_cfg.best_checkpoint_metric, float("nan")))
        is_best = best_val_score is None or val_score < best_val_score  # lower=better for loss
        if is_best:
            best_val_score = val_score

        if epoch % train_cfg.checkpoint_every == 0:
            ckpt_path = out_dir / f"epoch_{epoch:04d}.ckpt"
            save_beta_checkpoint(
                ckpt_path, model, optimizer, epoch, cfg,
                ema_model=ema_model,
                metrics=val_metrics,
            )
            print(f"  [epoch {epoch}] saved {ckpt_path.name}")

        if is_best:
            save_beta_checkpoint(
                out_dir / "best.ckpt", model, optimizer, epoch, cfg,
                ema_model=ema_model,
                metrics=val_metrics,
            )

        print(
            f"  [epoch {epoch}/{train_cfg.num_epochs}]"
            f"  train_loss={train_metrics['loss_termination']:.4f}"
            f"  val_loss={val_metrics['loss_termination']:.4f}"
            f"  beta_acc_pos={val_metrics.get('beta_acc_pos', float('nan')):.3f}"
            f"  beta_acc_neg={val_metrics.get('beta_acc_neg', float('nan')):.3f}"
            f"  gn[resnet/head/emb]="
            f"{train_metrics.get('grad_norm/obs_encoder', float('nan')):.3f}/"
            f"{train_metrics.get('grad_norm/term_head', float('nan')):.3f}/"
            f"{train_metrics.get('grad_norm/option_embedding', float('nan')):.3f}"
            f"  {'[BEST]' if is_best else ''}"
        )

    # Save final checkpoint
    save_beta_checkpoint(
        out_dir / "latest.ckpt", model, optimizer, train_cfg.num_epochs, cfg,
        ema_model=ema_model,
        metrics=val_metrics,
    )

    # Save metrics log
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_log, f, indent=2)

    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print(f"\n[train_beta] Done. Best val {train_cfg.best_checkpoint_metric}={best_val_score:.4f}")
    print(f"[train_beta] Checkpoints in: {out_dir}")


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    import hydra

    @hydra.main(config_path=None, config_name=None, version_base=None)
    def _main(cfg: Any) -> None:
        run_training(cfg)

    _main()


if __name__ == "__main__":
    import sys
    import hydra

    # Support: python soda/training/train_beta.py --config-path ... --config-name ...
    @hydra.main(config_path=None, config_name=None, version_base=None)
    def _main(cfg: Any) -> None:
        run_training(cfg)

    _main()
