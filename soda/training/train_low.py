"""
Train π_low (diffusion + termination) — project_plan §7 row 20.

Pairs with ``soda/models/low_policy.py`` (``LowPolicy.compute_loss``),
``soda/training/losses_low.py``, and ``OptionAwareDataset``.

Entrypoints
-----------
- Local: ``python soda/training/train_low.py --config-path configs/pusht --config-name soda_supervised``
- Modal: ``modal run modal/modal_train_low.py``
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from soda.dataset.option_aware_dataset import (
    OptionAwareDataset,
    build_option_dataset_from_config,
)
from soda.models.low_policy import LowPolicyConfig
from soda.models.termination_head import TerminationHeadConfig

__all__ = [
    "LowPolicyConfig",
    "TerminationHeadConfig",
    "TrainLowConfig",
    "build_datasets",
    "build_dataloaders",
    "build_policy_and_optimizer",
    "train_one_epoch",
    "validate",
    "save_checkpoint",
    "run_training",
    "main",
]


@dataclass
class TrainLowConfig:
    device: str = "cuda:0"
    seed: int = 42
    num_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-6
    checkpoint_every: int = 10
    output_dir: str | None = None
    wandb_enabled: bool = False
    wandb_project: str = "soda-train-low"
    finetune_dp_checkpoint: str | None = None
    num_workers: int = 4
    option_balance: str = "none"
    use_ema: bool = True
    lr_warmup_epochs: int = 5
    beta_stratified_batches: bool = True
    beta_batch_pos_fraction: float | None = None
    best_checkpoint_metric: str = "loss_diffusion"
    run_readme: str | None = None

    @classmethod
    def from_hydra(cls, cfg: Any) -> TrainLowConfig:
        block = _get_block(cfg, "train_low")
        return cls(
            device=str(_cfg_get(block, "device", "cuda:0")),
            seed=int(_cfg_get(block, "seed", 42)),
            num_epochs=int(_cfg_get(block, "num_epochs", 100)),
            batch_size=int(_cfg_get(block, "batch_size", 64)),
            lr=float(_cfg_get(block, "lr", 1e-4)),
            weight_decay=float(_cfg_get(block, "weight_decay", 1e-6)),
            checkpoint_every=int(_cfg_get(block, "checkpoint_every", 10)),
            output_dir=_cfg_get(block, "output_dir", None),
            wandb_enabled=bool(_cfg_get(block, "wandb_enabled", False)),
            wandb_project=str(_cfg_get(block, "wandb_project", "soda-train-low")),
            finetune_dp_checkpoint=_cfg_get(block, "finetune_dp_checkpoint", None),
            num_workers=int(_cfg_get(block, "num_workers", 4)),
            option_balance=str(_cfg_get(block, "option_balance", "none")),
            use_ema=bool(_cfg_get(block, "use_ema", True)),
            lr_warmup_epochs=int(_cfg_get(block, "lr_warmup_epochs", 5)),
            beta_stratified_batches=bool(_cfg_get(block, "beta_stratified_batches", True)),
            beta_batch_pos_fraction=_cfg_get(block, "beta_batch_pos_fraction", None),
            best_checkpoint_metric=str(
                _cfg_get(block, "best_checkpoint_metric", "loss_diffusion")
            ),
            run_readme=_cfg_get(block, "run_readme", None),
        )


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_block(cfg: Any, name: str) -> Any:
    return _cfg_get(cfg, name, {})


def _ensure_diffusion_policy() -> None:
    from soda.eval.policy_loaders import _ensure_diffusion_policy_path

    _ensure_diffusion_policy_path()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_hydra_cfg(cfg: Any, *, train: bool, random_anchor: bool) -> Any:
    """Build cfg slice for ``build_option_dataset_from_config``."""
    from omegaconf import OmegaConf

    ds = _cfg_get(_cfg_get(cfg, "task", {}), "dataset", {})
    ds_container = (
        OmegaConf.to_container(ds, resolve=True) if ds is not None else {}
    )
    if not isinstance(ds_container, dict):
        ds_container = {}
    merged = OmegaConf.create(
        {
            "dataset": OmegaConf.merge(
                OmegaConf.create(ds_container),
                {
                    "n_obs_steps": int(_cfg_get(cfg, "n_obs_steps", 2)),
                    "horizon": _cfg_get(cfg, "horizon", None),
                    "train": train,
                    "random_anchor": random_anchor,
                },
            )
        }
    )
    return merged


def pusht_shape_meta() -> dict[str, Any]:
    """Push-T hybrid layout (native action dim D=2; ``LowPolicy`` patches to D+1)."""
    return {
        "action": {"shape": [2]},
        "obs": {
            "image": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
    }


def _infer_num_options(dataset: OptionAwareDataset) -> int:
    ids = np.asarray(dataset._option_ids)
    return int(len(np.unique(ids)))


def _resolve_num_options(cfg: Any, dataset: OptionAwareDataset) -> int:
    configured = _cfg_get(_get_block(cfg, "low_policy"), "num_options", None)
    if configured is not None:
        return int(configured)
    return _infer_num_options(dataset)


def _low_policy_config(
    cfg: Any,
    dataset: OptionAwareDataset,
    *,
    termination_pos_weight: float,
) -> LowPolicyConfig:
    block = _get_block(cfg, "low_policy")
    th = _get_block(block, "termination_head")
    bottleneck_dim = int(_cfg_get(block, "bottleneck_dim", 1024))
    return LowPolicyConfig(
        num_options=_resolve_num_options(cfg, dataset),
        option_embed_dim=int(_cfg_get(block, "option_embed_dim", 32)),
        bottleneck_dim=bottleneck_dim,
        termination_head=TerminationHeadConfig(
            bottleneck_dim=int(_cfg_get(th, "bottleneck_dim", bottleneck_dim)),
            hidden_dim=int(_cfg_get(th, "hidden_dim", 256)),
            num_layers=int(_cfg_get(th, "num_layers", 2)),
        ),
        termination_loss_weight=float(_cfg_get(block, "termination_loss_weight", 1.0)),
        termination_pos_weight=float(termination_pos_weight),
        termination_input=str(_cfg_get(block, "termination_input", "bottleneck")),
        termination_stop_grad=bool(_cfg_get(block, "termination_stop_grad", True)),
    )


def _build_noise_scheduler(cfg: Any) -> Any:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    sched_cfg = dict(_get_block(cfg, "noise_scheduler") or {})
    defaults = {
        "num_train_timesteps": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "squaredcos_cap_v2",
        "clip_sample": True,
        "prediction_type": "epsilon",
        "variance_type": "fixed_small",
    }
    for key, value in defaults.items():
        sched_cfg.setdefault(key, value)
    return DDPMScheduler(**sched_cfg)


def build_datasets(cfg: Any) -> tuple[OptionAwareDataset, OptionAwareDataset]:
    train_ds = build_option_dataset_from_config(
        _dataset_hydra_cfg(cfg, train=True, random_anchor=True)
    )
    val_ds = train_ds.get_validation_dataset()
    return train_ds, val_ds


def build_dataloaders(
    cfg: Any,
    train_ds: OptionAwareDataset,
    val_ds: OptionAwareDataset,
    train_cfg: TrainLowConfig,
    *,
    beta_batch_pos_fraction: float | None = None,
) -> tuple[DataLoader, DataLoader, Any | None, Any | None]:
    from soda.dataset.option_stratified_sampler import OptionBetaStratifiedBatchSampler

    ds_cfg = _cfg_get(_cfg_get(cfg, "task", {}), "dataset", {})
    num_workers = int(_cfg_get(ds_cfg, "num_workers", train_cfg.num_workers))
    train_batch_sampler = None
    val_batch_sampler = None

    if train_cfg.beta_stratified_batches and beta_batch_pos_fraction is not None:
        train_ds.enable_stratified_indices(True)
        val_ds.enable_stratified_indices(True)
        train_batch_sampler = OptionBetaStratifiedBatchSampler(
            num_segments=len(train_ds),
            batch_size=train_cfg.batch_size,
            pos_fraction=beta_batch_pos_fraction,
            seed=train_cfg.seed,
            drop_last=True,
        )
        val_batch_sampler = OptionBetaStratifiedBatchSampler(
            num_segments=len(val_ds),
            batch_size=train_cfg.batch_size,
            pos_fraction=beta_batch_pos_fraction,
            seed=train_cfg.seed + 1,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=str(train_cfg.device).startswith("cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_batch_sampler,
            num_workers=num_workers,
            pin_memory=str(train_cfg.device).startswith("cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=str(train_cfg.device).startswith("cuda"),
            drop_last=True,
        )
    if val_batch_sampler is None:
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=str(train_cfg.device).startswith("cuda"),
            drop_last=False,
        )
    return train_loader, val_loader, train_batch_sampler, val_batch_sampler


def _build_low_policy(
    cfg: Any,
    dataset: OptionAwareDataset,
    *,
    termination_pos_weight: float,
) -> Any:
    from soda.models.low_policy import LowPolicy

    policy_cfg = _get_block(cfg, "policy")
    crop = _cfg_get(policy_cfg, "crop_shape", [84, 84])
    down_dims = _cfg_get(policy_cfg, "down_dims", [512, 1024, 2048])

    policy = LowPolicy(
        cfg=_low_policy_config(
            cfg, dataset, termination_pos_weight=termination_pos_weight
        ),
        shape_meta=pusht_shape_meta(),
        noise_scheduler=_build_noise_scheduler(cfg),
        horizon=int(dataset.horizon),
        n_action_steps=min(int(dataset.horizon), 8),
        n_obs_steps=int(_cfg_get(cfg, "n_obs_steps", 2)),
        unet_down_dims=tuple(int(x) for x in down_dims),
        unet_diffusion_step_embed_dim=int(
            _cfg_get(policy_cfg, "diffusion_step_embed_dim", 256)
        ),
        unet_kernel_size=int(_cfg_get(policy_cfg, "kernel_size", 5)),
        unet_n_groups=int(_cfg_get(policy_cfg, "n_groups", 8)),
        unet_cond_predict_scale=bool(_cfg_get(policy_cfg, "cond_predict_scale", True)),
        crop_shape=tuple(int(x) for x in crop),
        num_inference_steps=int(_cfg_get(policy_cfg, "num_inference_steps", 100)),
        eval_fixed_crop=bool(_cfg_get(policy_cfg, "eval_fixed_crop", False)),
        obs_encoder_group_norm=bool(
            _cfg_get(policy_cfg, "obs_encoder_group_norm", False)
        ),
        obs_as_global_cond=bool(_cfg_get(cfg, "obs_as_global_cond", True)),
    )
    return policy


def _load_finetune_weights(policy: Any, checkpoint: str | Path, device: str) -> None:
    from soda.eval.policy_loaders import load_dp_image_policy_and_cfg

    dp_policy, _, _ = load_dp_image_policy_and_cfg(checkpoint, device=device)
    missing, unexpected = policy.load_state_dict(dp_policy.state_dict(), strict=False)
    print(
        f"Finetune from DP checkpoint {checkpoint}: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def build_policy_and_optimizer(
    cfg: Any,
    train_dataset: OptionAwareDataset,
    train_cfg: TrainLowConfig,
    *,
    termination_pos_weight: float | None = None,
) -> tuple[Any, torch.optim.Optimizer]:
    _ensure_diffusion_policy()
    device = torch.device(train_cfg.device)

    if termination_pos_weight is None:
        termination_pos_weight = _resolve_termination_pos_weight(cfg, train_dataset)
    policy = _build_low_policy(
        cfg,
        train_dataset,
        termination_pos_weight=termination_pos_weight,
    )
    policy.set_normalizer(train_dataset.get_normalizer())

    if train_cfg.finetune_dp_checkpoint:
        _load_finetune_weights(policy, train_cfg.finetune_dp_checkpoint, train_cfg.device)

    policy.to(device)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=train_cfg.lr,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=train_cfg.weight_decay,
    )
    return policy, optimizer


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    from diffusion_policy.common.pytorch_util import dict_apply

    return dict_apply(
        batch,
        lambda x: x.to(device, non_blocking=True) if torch.is_tensor(x) else x,
    )


def _aggregate_logs(log_keys: list[str], sums: dict[str, float], count: int) -> dict[str, float]:
    if count == 0:
        return {k: float("nan") for k in log_keys}
    return {k: sums[k] / count for k in log_keys}


def _resolve_termination_pos_weight(
    cfg: Any,
    train_dataset: OptionAwareDataset,
) -> float:
    from soda.dataset.option_aware_dataset import episode_train_mask
    from soda.training.option_balance import (
        format_termination_pos_weight_summary,
        resolve_termination_pos_weight,
    )

    block = _get_block(cfg, "low_policy")
    configured = _cfg_get(block, "termination_pos_weight", None)
    train_ep_mask = episode_train_mask(
        int(train_dataset._episode_ends.size),
        val_ratio=train_dataset._val_ratio,
        seed=train_dataset._seed,
        max_train_episodes=train_dataset._max_train_episodes,
    )
    gamma, n_neg, n_pos, manual = resolve_termination_pos_weight(
        configured=configured,
        option_ids=train_dataset._option_ids,
        episode_ends=train_dataset._episode_ends,
        train_episode_mask=train_ep_mask,
    )
    print(
        format_termination_pos_weight_summary(
            gamma=gamma,
            n_neg=n_neg,
            n_pos=n_pos,
            manual=manual,
        )
    )
    return gamma


def _resolve_option_class_weights(
    cfg: Any,
    train_dataset: OptionAwareDataset,
) -> torch.Tensor | None:
    from soda.training.option_balance import (
        count_option_ids,
        option_ids_from_segments,
        resolve_option_class_weights,
    )

    num_options = _resolve_num_options(cfg, train_dataset)
    mode = str(_cfg_get(_get_block(cfg, "train_low"), "option_balance", "none"))
    option_ids = option_ids_from_segments(train_dataset.segments)
    weights = resolve_option_class_weights(mode, option_ids, num_options)
    if weights is not None:
        counts = count_option_ids(option_ids, num_options)
        skill_names = None
        try:
            from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES

            skill_names = SKILL_NAMES
        except ImportError:
            pass
        from soda.training.option_balance import format_option_weight_summary

        print(format_option_weight_summary(counts, weights, skill_names=skill_names))
    return weights


def train_one_epoch(
    policy: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    ema: Any | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    policy.train()
    torch_device = torch.device(device)
    log_keys = ("loss", "loss_diffusion", "loss_termination")
    sums = {k: 0.0 for k in log_keys}
    n_batches = 0
    weights_on_device = (
        class_weights.to(torch_device) if class_weights is not None else None
    )

    for batch in tqdm(loader, desc="train", leave=False):
        batch = _move_batch_to_device(batch, torch_device)
        loss, logs = policy.compute_loss(batch, class_weights=weights_on_device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.step(policy)
        for key in log_keys:
            sums[key] += float(logs[key])
        n_batches += 1

    return _aggregate_logs(log_keys, sums, n_batches)


@torch.no_grad()
def validate(
    policy: Any,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    policy.eval()
    torch_device = torch.device(device)
    log_keys = ("loss", "loss_diffusion", "loss_termination")
    sums = {k: 0.0 for k in log_keys}
    n_batches = 0
    from soda.training.losses_low import (
        BetaConfusionCounts,
        beta_confusion_counts_from_logits,
        beta_metrics_from_counts,
    )

    beta_counts = BetaConfusionCounts()

    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch_to_device(batch, torch_device)
        _, logs = policy.compute_loss(batch)
        for key in log_keys:
            sums[key] += float(logs[key])
        n_batches += 1

        beta_logit = policy.beta_logits_from_batch(batch)
        beta_counts += beta_confusion_counts_from_logits(
            beta_logit,
            batch["beta_label"],
            threshold=0.5,
            high_prob_threshold=0.9,
        )

    metrics = _aggregate_logs(log_keys, sums, n_batches)
    metrics.update(beta_metrics_from_counts(beta_counts, high_prob_threshold=0.9))
    return metrics


def _checkpoint_score(val_metrics: dict[str, float], metric_key: str) -> float:
    """Lower is better for loss metrics; higher is better for accuracy metrics."""
    if metric_key not in val_metrics:
        raise KeyError(
            f"best_checkpoint_metric={metric_key!r} missing from val metrics "
            f"(keys: {sorted(val_metrics)})"
        )
    value = float(val_metrics[metric_key])
    if metric_key.startswith("beta_") and "acc" in metric_key:
        return -value
    if metric_key.startswith("beta_precision") or metric_key.startswith("beta_recall"):
        return -value
    return value


def build_low_policy_from_saved_config(
    hydra_cfg: Any,
    lp_cfg: LowPolicyConfig,
    *,
    horizon: int,
) -> Any:
    """Rebuild π_low for eval reload when ``low_policy_config`` is stored in the ckpt."""
    _ensure_diffusion_policy()
    from soda.models.low_policy import LowPolicy

    policy_cfg = _get_block(hydra_cfg, "policy")
    crop = _cfg_get(policy_cfg, "crop_shape", [84, 84])
    down_dims = _cfg_get(policy_cfg, "down_dims", [512, 1024, 2048])
    return LowPolicy(
        cfg=lp_cfg,
        shape_meta=pusht_shape_meta(),
        noise_scheduler=_build_noise_scheduler(hydra_cfg),
        horizon=int(horizon),
        n_action_steps=int(_cfg_get(hydra_cfg, "n_action_steps", 8)),
        n_obs_steps=int(_cfg_get(hydra_cfg, "n_obs_steps", 2)),
        unet_down_dims=tuple(int(x) for x in down_dims),
        unet_diffusion_step_embed_dim=int(
            _cfg_get(policy_cfg, "diffusion_step_embed_dim", 256)
        ),
        unet_kernel_size=int(_cfg_get(policy_cfg, "kernel_size", 5)),
        unet_n_groups=int(_cfg_get(policy_cfg, "n_groups", 8)),
        unet_cond_predict_scale=bool(_cfg_get(policy_cfg, "cond_predict_scale", True)),
        crop_shape=tuple(int(x) for x in crop),
        num_inference_steps=int(_cfg_get(policy_cfg, "num_inference_steps", 100)),
        eval_fixed_crop=bool(_cfg_get(policy_cfg, "eval_fixed_crop", False)),
        obs_encoder_group_norm=bool(
            _cfg_get(policy_cfg, "obs_encoder_group_norm", False)
        ),
        obs_as_global_cond=bool(_cfg_get(hydra_cfg, "obs_as_global_cond", True)),
    )


def save_checkpoint(
    path: Path,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Any,
    *,
    ema_policy: Any | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    from dataclasses import asdict

    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "policy": policy.state_dict(),
        "ema_policy": ema_policy.state_dict() if ema_policy is not None else None,
        "optimizer": optimizer.state_dict(),
        "normalizer": policy.normalizer.state_dict(),
        "low_policy_config": asdict(policy.cfg),
        "horizon": int(policy.horizon),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def _resolve_output_dir(cfg: Any, train_cfg: TrainLowConfig) -> Path:
    if train_cfg.output_dir:
        return Path(train_cfg.output_dir)
    name = str(_cfg_get(cfg, "name", "soda_train_low"))
    return Path("experiments") / "train_low" / name


def run_training(cfg: Any) -> None:
    from omegaconf import OmegaConf

    from soda.experiments.run_readme import (
        begin_training_run_archive,
        finalize_training_run_archive,
    )

    train_cfg = TrainLowConfig.from_hydra(cfg)
    _seed_all(train_cfg.seed)
    out_dir = _resolve_output_dir(cfg, train_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_archive = begin_training_run_archive(
        out_dir,
        kind="train_low",
        hydra_readme=train_cfg.run_readme,
        extra_metadata={
            "config_name": str(_cfg_get(cfg, "name", out_dir.name)),
            "num_epochs": train_cfg.num_epochs,
        },
    )

    train_ds, val_ds = build_datasets(cfg)
    termination_pos_weight = _resolve_termination_pos_weight(cfg, train_ds)
    beta_batch_pos_fraction = None
    if train_cfg.beta_stratified_batches:
        from soda.dataset.option_stratified_sampler import positives_per_batch
        from soda.training.option_balance import resolve_beta_batch_pos_fraction

        beta_batch_pos_fraction = resolve_beta_batch_pos_fraction(
            train_cfg.beta_batch_pos_fraction,
            termination_pos_weight=termination_pos_weight,
        )
        n_pos = positives_per_batch(train_cfg.batch_size, beta_batch_pos_fraction)
        print(
            "Beta stratified batches (train + val): "
            f"pos_fraction={beta_batch_pos_fraction:.4f} "
            f"({n_pos}/{train_cfg.batch_size} beta=1 per batch, "
            f"auto=1/(gamma+1) when beta_batch_pos_fraction null; "
            f"val uses fixed epoch-0 shuffle)"
        )

    train_loader, val_loader, train_batch_sampler, val_batch_sampler = build_dataloaders(
        cfg,
        train_ds,
        val_ds,
        train_cfg,
        beta_batch_pos_fraction=beta_batch_pos_fraction,
    )
    policy, optimizer = build_policy_and_optimizer(
        cfg,
        train_ds,
        train_cfg,
        termination_pos_weight=termination_pos_weight,
    )
    class_weights = _resolve_option_class_weights(cfg, train_ds)

    # EMA — same config as Columbia upstream (inv_gamma=1, power=0.75, max=0.9999)
    ema_model = None
    ema = None
    if train_cfg.use_ema:
        _ensure_diffusion_policy()
        from diffusion_policy.model.diffusion.ema_model import EMAModel

        ema_model = copy.deepcopy(policy)
        ema_model.to(torch.device(train_cfg.device))
        ema = EMAModel(
            model=ema_model,
            update_after_step=0,
            inv_gamma=1.0,
            power=0.75,
            min_value=0.0,
            max_value=0.9999,
        )

    # LR scheduler: linear warmup → cosine decay (per-epoch, robust to changing num_epochs)
    warmup_epochs = min(train_cfg.lr_warmup_epochs, train_cfg.num_epochs)
    cosine_epochs = max(1, train_cfg.num_epochs - warmup_epochs)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=0.0
            ),
        ],
        milestones=[warmup_epochs],
    )

    print(
        f"π_low train: {len(train_ds)} segments (train), "
        f"{len(val_ds)} segments (val), "
        f"horizon={train_ds.horizon}, "
        f"num_options={_resolve_num_options(cfg, train_ds)}, "
        f"option_balance={train_cfg.option_balance}, "
        f"use_ema={train_cfg.use_ema}, "
        f"lr_warmup_epochs={warmup_epochs}, "
        f"best_checkpoint_metric={train_cfg.best_checkpoint_metric}, "
        f"termination_input={policy.cfg.termination_input}, "
        f"termination_stop_grad={policy.cfg.termination_stop_grad}"
    )

    wandb_run = None
    if train_cfg.wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=train_cfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(out_dir),
        )

    best_val = float("inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, train_cfg.num_epochs + 1):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        if val_batch_sampler is not None:
            val_batch_sampler.set_epoch(0)
        train_metrics = train_one_epoch(
            policy,
            train_loader,
            optimizer,
            train_cfg.device,
            ema=ema,
            class_weights=class_weights,
        )
        current_lr = optimizer.param_groups[0]['lr']
        lr_scheduler.step()
        eval_policy = ema_model if ema_model is not None else policy
        val_metrics = validate(eval_policy, val_loader, train_cfg.device)
        row = {"epoch": epoch, "lr": current_lr,
               **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row))

        if wandb_run is not None:
            import wandb

            wandb.log(row, step=epoch)

        if _checkpoint_score(val_metrics, train_cfg.best_checkpoint_metric) < best_val:
            best_val = _checkpoint_score(val_metrics, train_cfg.best_checkpoint_metric)
            save_checkpoint(
                out_dir / "best.ckpt",
                policy,
                optimizer,
                epoch,
                cfg,
                ema_policy=ema_model,
                metrics=val_metrics,
            )

        if epoch % train_cfg.checkpoint_every == 0 or epoch == train_cfg.num_epochs:
            save_checkpoint(
                out_dir / f"epoch_{epoch:04d}.ckpt",
                policy,
                optimizer,
                epoch,
                cfg,
                ema_policy=ema_model,
                metrics=val_metrics,
            )

    save_checkpoint(
        out_dir / "latest.ckpt",
        policy,
        optimizer,
        train_cfg.num_epochs,
        cfg,
        ema_policy=ema_model,
        metrics=history[-1] if history else {},
    )
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if wandb_run is not None:
        import wandb

        wandb.finish()

    finalize_training_run_archive(run_archive)


def main(cfg: Any) -> None:
    """Hydra entrypoint — called by CLI wrapper in ``__main__``."""
    run_training(cfg)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        version_base=None,
        config_path="../../configs/pusht",
        config_name="soda_supervised",
    )
    def _cli(cfg: DictConfig) -> None:
        main(cfg)

    _cli()
