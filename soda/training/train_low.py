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
import os
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

# ---------------------------------------------------------------------------
# DDP helpers (safe to call even when torch.distributed is not initialized)
# ---------------------------------------------------------------------------

def _ddp_is_active() -> bool:
    """True when launched under torchrun (LOCAL_RANK env var is set)."""
    return "LOCAL_RANK" in os.environ


def _ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _ddp_init(device_str: str) -> tuple[int, int, str]:
    """Initialize NCCL process group. Returns (rank, world_size, per-rank device string)."""
    import torch.distributed as dist

    local_rank = _ddp_local_rank()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"


def _ddp_cleanup() -> None:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def _ddp_barrier() -> None:
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
    except Exception:
        pass


def _ddp_reduce_epoch_metrics(sums: dict[str, float], n_batches: int) -> tuple[dict[str, float], int]:
    """All-reduce epoch sums + batch count across DDP ranks, returning global means."""
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return sums, n_batches
        keys = sorted(sums.keys())
        t = torch.tensor(
            [sums[k] for k in keys] + [float(n_batches)],
            dtype=torch.float64,
            device="cuda",
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced = {k: float(t[i]) for i, k in enumerate(keys)}
        return reduced, int(round(float(t[-1])))
    except Exception:
        return sums, n_batches


def _ddp_sync_gradients(params: list) -> None:
    """All-reduce gradients across DDP ranks (replaces DDP auto-bucketing for custom training loops)."""
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        ws = dist.get_world_size()
        grads = [p.grad.data for p in params if p.grad is not None]
        if not grads:
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(ws)
        offset = 0
        for g in grads:
            n = g.numel()
            g.copy_(flat[offset:offset + n].reshape(g.shape))
            offset += n
    except Exception:
        pass


def _unwrap_policy(policy: Any) -> Any:
    """Unwrap DDP wrapper; no-op for plain modules."""
    if hasattr(policy, "module"):
        return policy.module
    return policy


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
    # --- Shared infra ---
    device: str
    seed: int
    batch_size: int
    weight_decay: float
    checkpoint_every: int
    output_dir: str | None           # null = auto (experiments/train_low/{name}/)
    wandb_enabled: bool
    wandb_project: str
    wandb_run_name: str | None       # null = auto
    wandb_group: str | None          # null = no group
    wandb_tags: tuple[str, ...]
    finetune_dp_checkpoint: str | None  # null = no warmstart from Columbia DP checkpoint
    num_workers: int
    option_balance: str              # "none" | "inverse_freq"
    use_ema: bool
    lr_schedule_type: str            # "constant" | "cosine" | "plateau"
    lr_plateau_patience: int | None
    lr_plateau_patience_beta: int | None  # null = same as lr_plateau_patience
    lr_plateau_factor: float | None
    lr_plateau_min_lr: float | None
    beta_stratified_batches: bool
    run_readme: str | None
    # --- Training params ---
    num_epochs: int
    lr: float
    lr_warmup_epochs: int
    beta_lr: float | None            # null = same as lr
    warmstart_unet: bool | None      # null/true = full DP warmstart; false = obs_encoder only
    mask_diffusion_on_positive: bool
    best_checkpoint_metric: str      # "loss_diffusion" | "loss_termination"
    all_anchors: bool
    termination_loss_weight: float   # λ: scale on β BCE loss
    termination_pos_weight: float | None  # γ: BCE pos_weight; None → auto from data
    skip_diffusion_loss: bool        # skip noisy U-Net forward; backbone lr is set to 0 when true
    termination_stop_grad: bool      # detach β features from backbone
    checkpoint: str | None           # SODA warmstart checkpoint (null = start fresh)

    @classmethod
    def from_hydra(cls, cfg: Any) -> TrainLowConfig:
        block = _get_block(cfg, "train_low")

        _plat_beta_raw = _cfg_require(block, "lr_plateau_patience_beta", "train_low")
        plat_beta = int(_plat_beta_raw) if _plat_beta_raw is not None else None

        def _opt_float(v: Any) -> float | None:
            return None if v is None else float(v)

        def _opt_bool(v: Any) -> bool | None:
            return None if v is None else bool(v)

        return cls(
            device=str(_cfg_require(block, "device", "train_low")),
            seed=int(_cfg_require(block, "seed", "train_low")),
            batch_size=int(_cfg_require(block, "batch_size", "train_low")),
            weight_decay=float(_cfg_require(block, "weight_decay", "train_low")),
            checkpoint_every=int(_cfg_require(block, "checkpoint_every", "train_low")),
            output_dir=_cfg_get(block, "output_dir", None),
            wandb_enabled=bool(_cfg_require(block, "wandb_enabled", "train_low")),
            wandb_project=str(_cfg_require(block, "wandb_project", "train_low")),
            wandb_run_name=_cfg_get(block, "wandb_run_name", None),
            wandb_group=_cfg_get(block, "wandb_group", None),
            wandb_tags=_parse_wandb_tags(_cfg_get(block, "wandb_tags", None)),
            finetune_dp_checkpoint=_cfg_get(block, "finetune_dp_checkpoint", None),
            num_workers=int(_cfg_require(block, "num_workers", "train_low")),
            option_balance=str(_cfg_require(block, "option_balance", "train_low")),
            use_ema=bool(_cfg_require(block, "use_ema", "train_low")),
            lr_schedule_type=str(_cfg_require(block, "lr_schedule_type", "train_low")),
            lr_plateau_patience=(lambda v: int(v) if v is not None else None)(_cfg_require(block, "lr_plateau_patience", "train_low")),
            lr_plateau_patience_beta=plat_beta,
            lr_plateau_factor=_opt_float(_cfg_require(block, "lr_plateau_factor", "train_low")),
            lr_plateau_min_lr=_opt_float(_cfg_require(block, "lr_plateau_min_lr", "train_low")),
            beta_stratified_batches=bool(_cfg_require(block, "beta_stratified_batches", "train_low")),
            run_readme=_cfg_get(block, "run_readme", None),
            num_epochs=int(_cfg_require(block, "num_epochs", "train_low")),
            lr=float(_cfg_require(block, "lr", "train_low")),
            lr_warmup_epochs=int(_cfg_get(block, "lr_warmup_epochs", 0)),
            beta_lr=_opt_float(_cfg_get(block, "beta_lr", None)),
            warmstart_unet=_opt_bool(_cfg_get(block, "warmstart_unet", None)),
            mask_diffusion_on_positive=bool(_cfg_require(block, "mask_diffusion_on_positive", "train_low")),
            best_checkpoint_metric=str(_cfg_require(block, "best_checkpoint_metric", "train_low")),
            all_anchors=bool(_cfg_require(block, "all_anchors", "train_low")),
            termination_loss_weight=float(_cfg_require(block, "termination_loss_weight", "train_low")),
            termination_pos_weight=_opt_float(_cfg_require(block, "termination_pos_weight", "train_low")),
            skip_diffusion_loss=bool(_cfg_require(block, "skip_diffusion_loss", "train_low")),
            termination_stop_grad=bool(_cfg_require(block, "termination_stop_grad", "train_low")),
            checkpoint=(lambda v: str(v) if v is not None else None)(_cfg_get(block, "checkpoint", None)),
        )


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_MISSING = object()


def _cfg_require(cfg: Any, key: str, context: str) -> Any:
    """Return ``cfg[key]`` or raise a descriptive error if the key is absent.

    Unlike ``_cfg_get``, this never falls back to a default — the key must be
    explicitly present in the YAML.  ``null`` is a valid value (returned as-is);
    only *absence* triggers the error.
    """
    if isinstance(cfg, dict):
        if key not in cfg:
            raise ValueError(
                f"Required config key '{key}' is missing in '{context}'. "
                f"Set it explicitly in the YAML (use null for optional fields)."
            )
        return cfg[key]
    val = getattr(cfg, key, _MISSING)
    if val is _MISSING:
        raise ValueError(
            f"Required config key '{key}' is missing in '{context}'. "
            f"Set it explicitly in the YAML (use null for optional fields)."
        )
    return val


def _get_block(cfg: Any, name: str) -> Any:
    return _cfg_get(cfg, name, {})


def _parse_wandb_tags(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return tuple(str(part) for part in raw)




def _init_wandb_run(train_cfg: TrainLowConfig, cfg: Any, out_dir: Path) -> Any:
    from omegaconf import OmegaConf

    import wandb

    init_kwargs: dict[str, Any] = {
        "project": train_cfg.wandb_project,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dir": str(out_dir),
    }
    if train_cfg.wandb_run_name:
        init_kwargs["name"] = str(train_cfg.wandb_run_name)
    if train_cfg.wandb_group:
        init_kwargs["group"] = str(train_cfg.wandb_group)
    if train_cfg.wandb_tags:
        init_kwargs["tags"] = list(train_cfg.wandb_tags)
    return wandb.init(**init_kwargs)


def _ensure_diffusion_policy() -> None:
    from soda.eval.policy_loaders import _ensure_diffusion_policy_path

    _ensure_diffusion_policy_path()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_hydra_cfg(cfg: Any, *, train: bool, random_anchor: bool, all_anchors: bool = False) -> Any:
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
                    "all_anchors": all_anchors,
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
    termination_loss_weight: float,
    termination_pos_weight: float | None,
) -> LowPolicyConfig:
    block = _get_block(cfg, "low_policy")
    th = _get_block(block, "termination_head")
    bottleneck_dim = int(_cfg_get(block, "bottleneck_dim", 1024))
    # Resolve pos_weight from data if null (auto-calculation accounts for escape_relabeling).
    resolved_pos_weight = (
        _resolve_termination_pos_weight(cfg, dataset, configured=termination_pos_weight)
        if termination_pos_weight is None
        else float(termination_pos_weight)
    )
    return LowPolicyConfig(
        num_options=_resolve_num_options(cfg, dataset),
        option_embed_dim=int(_cfg_get(block, "option_embed_dim", 32)),
        bottleneck_dim=bottleneck_dim,
        termination_head=TerminationHeadConfig(
            bottleneck_dim=int(_cfg_get(th, "bottleneck_dim", bottleneck_dim)),
            hidden_dim=int(_cfg_get(th, "hidden_dim", 256)),
            num_layers=int(_cfg_get(th, "num_layers", 2)),
            use_chunk_cursor=bool(_cfg_get(th, "use_chunk_cursor", False)),
            dropout_rate=float(_cfg_get(th, "dropout_rate", 0.0)),
        ),
        termination_loss_weight=float(termination_loss_weight),
        termination_pos_weight=resolved_pos_weight,
        termination_input=str(_cfg_get(block, "termination_input", "bottleneck")),
        termination_stop_grad=bool(_cfg_get(block, "termination_stop_grad", True)),
        termination_label_smoothing=float(_cfg_get(block, "termination_label_smoothing", 0.0)),
        imagenet_init=bool(_cfg_get(block, "imagenet_init", False)),
        escape_relabeling=bool(_cfg_get(block, "escape_relabeling", False)),
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


def build_datasets(cfg: Any, *, all_anchors: bool = False) -> tuple[OptionAwareDataset, OptionAwareDataset]:
    escape_relabeling = bool(_cfg_get(_get_block(cfg, "low_policy"), "escape_relabeling", False))
    ds_cfg = _dataset_hydra_cfg(cfg, train=True, random_anchor=True, all_anchors=all_anchors)
    # Inject escape_relabeling into the dataset cfg so build_option_dataset_from_config picks it up.
    from omegaconf import OmegaConf
    ds_cfg = OmegaConf.merge(ds_cfg, OmegaConf.create({"dataset": {"escape_relabeling": escape_relabeling}}))
    train_ds = build_option_dataset_from_config(ds_cfg)
    val_ds = train_ds.get_validation_dataset(all_anchors=all_anchors)
    return train_ds, val_ds


def build_dataloaders(
    cfg: Any,
    train_ds: OptionAwareDataset,
    val_ds: OptionAwareDataset,
    train_cfg: TrainLowConfig,
    *,
    beta_batch_pos_fraction: float | None = None,
    all_anchors: bool = False,
) -> tuple[DataLoader, DataLoader, Any | None, Any | None]:
    from soda.dataset.option_stratified_sampler import (
        AllAnchorsBetaStratifiedBatchSampler,
        OptionBetaStratifiedBatchSampler,
        positives_per_batch,
    )

    ds_cfg = _cfg_get(_cfg_get(cfg, "task", {}), "dataset", {})
    num_workers = int(_cfg_get(ds_cfg, "num_workers", train_cfg.num_workers))
    train_batch_sampler = None
    val_batch_sampler = None
    pin = str(train_cfg.device).startswith("cuda")

    if all_anchors:
        import torch.distributed as dist
        _is_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        if _is_ddp:
            from torch.utils.data.distributed import DistributedSampler
            train_batch_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
            train_loader = DataLoader(
                train_ds,
                batch_size=train_cfg.batch_size,
                sampler=train_batch_sampler,
                num_workers=num_workers,
                pin_memory=pin,
            )
            _val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
            val_loader = DataLoader(
                val_ds,
                batch_size=train_cfg.batch_size,
                sampler=_val_sampler,
                num_workers=num_workers,
                pin_memory=pin,
            )
            print(
                f"AllAnchors DDP: {len(train_ds)} train / {len(val_ds)} val anchors "
                f"(rank {dist.get_rank()}/{dist.get_world_size()}, "
                f"~{len(train_ds) // dist.get_world_size()} train per rank)"
            )
        else:
            # Linear beta labels give gradient at every anchor — uniform shuffle suffices.
            print(
                f"AllAnchors uniform batches: {len(train_ds)} train / {len(val_ds)} val anchors"
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=train_cfg.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin,
                drop_last=False,
            )
    elif train_cfg.beta_stratified_batches and beta_batch_pos_fraction is not None:
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
            pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_batch_sampler,
            num_workers=num_workers,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            drop_last=False,
        )
    return train_loader, val_loader, train_batch_sampler, val_batch_sampler


def _build_low_policy(
    cfg: Any,
    dataset: OptionAwareDataset,
    *,
    termination_loss_weight: float,
    termination_pos_weight: float,
) -> Any:
    from soda.models.low_policy import LowPolicy

    policy_cfg = _get_block(cfg, "policy")
    crop = _cfg_get(policy_cfg, "crop_shape", [84, 84])
    down_dims = _cfg_get(policy_cfg, "down_dims", [512, 1024, 2048])

    policy = LowPolicy(
        cfg=_low_policy_config(
            cfg, dataset,
            termination_loss_weight=termination_loss_weight,
            termination_pos_weight=termination_pos_weight,
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


def _load_finetune_weights(
    policy: Any,
    checkpoint: str | Path,
    device: str,
    *,
    warmstart_unet: bool | None = None,
) -> None:
    from soda.eval.policy_loaders import load_dp_image_policy_and_cfg

    dp_policy, _, _ = load_dp_image_policy_and_cfg(checkpoint, device=device)
    dp_state = dp_policy.state_dict()

    if warmstart_unet is False:
        # obs_encoder-only: load just the ResNet vision encoder, leave U-Net at random init.
        # This avoids the LR conflict where the duration output channel (inside policy.model)
        # cannot share a single LR with the pretrained U-Net action weights.
        obs_enc_state = {
            k.removeprefix("obs_encoder."): v
            for k, v in dp_state.items()
            if k.startswith("obs_encoder.")
        }
        policy.obs_encoder.load_state_dict(obs_enc_state, strict=True)
        print(
            f"Finetune from DP checkpoint {checkpoint}: "
            f"obs_encoder-only ({len(obs_enc_state)} keys); U-Net/option_embed/β at random init"
        )
        return

    # Full compatible warmstart (warmstart_unet=True or None → original behavior).
    # Filter to keys present in both with identical shapes; exclude normalizer (action dim differs).
    policy_state = policy.state_dict()
    compatible = {
        k: v for k, v in dp_state.items()
        if k in policy_state
        and v.shape == policy_state[k].shape
        and not k.startswith("normalizer.")
    }
    skipped = [k for k in dp_state if k not in compatible]
    missing, _ = policy.load_state_dict(compatible, strict=False)
    print(
        f"Finetune from DP checkpoint {checkpoint}: "
        f"transferred={len(compatible)}, skipped={len(skipped)}, missing={len(missing)}"
    )


def build_policy_and_optimizer(
    cfg: Any,
    train_dataset: OptionAwareDataset,
    train_cfg: TrainLowConfig,
    *,
    termination_loss_weight: float,
    termination_pos_weight: float,
) -> tuple[Any, torch.optim.Optimizer, torch.optim.Optimizer]:
    """Return (policy, optimizer_backbone, optimizer_beta).

    Two separate optimizers so each component gets an independent LR schedule:
    - optimizer_backbone: U-Net + obs encoder (all non-termination-head params)
    - optimizer_beta: termination head MLP only
    With stop_grad=true the gradient paths are already decoupled; separate optimizers
    let each LR schedule respond to its own convergence signal.
    """
    _ensure_diffusion_policy()
    device = torch.device(train_cfg.device)

    policy = _build_low_policy(
        cfg,
        train_dataset,
        termination_loss_weight=termination_loss_weight,
        termination_pos_weight=termination_pos_weight,
    )
    if train_cfg.finetune_dp_checkpoint:
        _load_finetune_weights(
            policy,
            train_cfg.finetune_dp_checkpoint,
            train_cfg.device,
            warmstart_unet=train_cfg.warmstart_unet,
        )

    # set_normalizer must come AFTER _load_finetune_weights: DictOfTensorMixin._load_from_state_dict
    # completely replaces params_dict from the incoming state dict, so any prior set_normalizer
    # call gets wiped even if normalizer keys are excluded from the transfer.
    policy.set_normalizer(train_dataset.get_normalizer())

    policy.to(device)

    backbone_params = [p for n, p in policy.named_parameters() if "termination_head" not in n]
    beta_params = [p for n, p in policy.named_parameters() if "termination_head" in n]
    beta_lr = train_cfg.beta_lr if train_cfg.beta_lr is not None else train_cfg.lr
    # When skip_diffusion_loss=True the backbone receives no gradients; set lr=0 so
    # AdamW weight decay also does not nudge frozen weights.
    backbone_lr = 0.0 if train_cfg.skip_diffusion_loss else train_cfg.lr

    _adamw_kwargs: dict = {"betas": (0.95, 0.999), "eps": 1e-8, "weight_decay": train_cfg.weight_decay}
    optimizer_backbone = torch.optim.AdamW(backbone_params, lr=backbone_lr, **_adamw_kwargs)
    optimizer_beta = torch.optim.AdamW(beta_params, lr=beta_lr, **_adamw_kwargs)

    n_backbone = sum(p.numel() for p in backbone_params)
    n_beta = sum(p.numel() for p in beta_params)
    print(f"Optimizers: backbone {n_backbone:,} params (lr={backbone_lr}), β head {n_beta:,} params (lr={beta_lr})")

    return policy, optimizer_backbone, optimizer_beta


def _build_ema(policy: Any, train_cfg: "TrainLowConfig") -> tuple[Any | None, Any | None]:
    if not train_cfg.use_ema:
        return None, None
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
    return ema_model, ema


def _reload_phase_checkpoint(policy: Any, ema_model: Any | None, checkpoint_path: str, device: str) -> None:
    """Load policy (and EMA) weights from a checkpoint without touching normalizer or optimizers.

    Filters out keys whose shapes don't match the current model (e.g. termination_head when
    switching termination_input mode between phases). Those keys are re-initialized from scratch,
    which is correct since phase 1 never trains the termination head (termination_loss_weight=0).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    def _filtered_load(model: Any, state: dict) -> None:
        current = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
        skipped = [k for k in state if k not in compatible]
        if skipped:
            print(f"  Checkpoint reload: skipped {len(skipped)} keys with incompatible shapes "
                  f"(termination_input mode change?): {skipped[:3]}{'...' if len(skipped) > 3 else ''}")
        missing, _ = model.load_state_dict(compatible, strict=False)
        if missing:
            print(f"  Checkpoint reload: {len(missing)} keys missing → random init")

    _filtered_load(policy, ckpt["policy"])
    if ema_model is not None and ckpt.get("ema_policy") is not None:
        _filtered_load(ema_model, ckpt["ema_policy"])
    print(f"Reloaded checkpoint for phase 2: {checkpoint_path}")


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
    configured: float | None = _MISSING,  # type: ignore[assignment]
) -> float:
    """Resolve pos_weight from config or auto-calculate from dataset class frequencies.

    Pass ``configured`` explicitly to override reading from cfg (e.g. when a phase
    has a phase-specific value). If omitted, reads ``low_policy.termination_pos_weight``
    from cfg; null → auto-calculate.
    """
    from soda.dataset.option_aware_dataset import episode_train_mask
    from soda.training.option_balance import (
        format_termination_pos_weight_summary,
        resolve_termination_pos_weight,
    )

    if configured is _MISSING:
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
        escape_relabeling=train_dataset.escape_relabeling,
        num_options=train_dataset.num_options,
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
    optimizer_backbone: torch.optim.Optimizer,
    optimizer_beta: torch.optim.Optimizer,
    device: str,
    *,
    ema: Any | None = None,
    class_weights: torch.Tensor | None = None,
    mask_diffusion_on_positive: bool = False,
    skip_diffusion: bool = False,
) -> dict[str, float]:
    policy.train()
    torch_device = torch.device(device)
    log_keys = ("loss", "loss_diffusion", "loss_termination")
    sums = {k: 0.0 for k in log_keys}
    n_batches = 0
    weights_on_device = (
        class_weights.to(torch_device) if class_weights is not None else None
    )
    # Collect param lists once for grad clipping (avoids repeated list comprehension per batch).
    backbone_params = [p for pg in optimizer_backbone.param_groups for p in pg["params"]]
    beta_params = [p for pg in optimizer_beta.param_groups for p in pg["params"]]

    for batch in tqdm(loader, desc="train", leave=False):
        batch = _move_batch_to_device(batch, torch_device)
        loss, logs = policy.compute_loss(
            batch,
            class_weights=weights_on_device,
            mask_diffusion_on_positive=mask_diffusion_on_positive,
            skip_diffusion=skip_diffusion,
        )
        optimizer_backbone.zero_grad(set_to_none=True)
        optimizer_beta.zero_grad(set_to_none=True)
        loss.backward()
        _ddp_sync_gradients(backbone_params + beta_params)
        # Clip each component independently — β head gradient can't shrink backbone updates.
        torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(beta_params, max_norm=1.0)
        optimizer_backbone.step()
        optimizer_beta.step()
        if ema is not None:
            ema.step(_unwrap_policy(policy))
        for key in log_keys:
            sums[key] += float(logs[key])
        n_batches += 1

    sums, n_batches = _ddp_reduce_epoch_metrics(sums, n_batches)
    return _aggregate_logs(log_keys, sums, n_batches)


@torch.no_grad()
def validate(
    policy: Any,
    loader: DataLoader,
    device: str,
    *,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    policy.eval()
    torch_device = torch.device(device)
    log_keys = ("loss", "loss_diffusion", "loss_termination")
    sums = {k: 0.0 for k in log_keys}
    n_batches = 0
    weights_on_device = class_weights.to(torch_device) if class_weights is not None else None
    from soda.training.losses_low import (
        BetaConfusionCounts,
        beta_confusion_counts_from_logits,
        beta_mae_from_logits,
        beta_metrics_from_counts,
    )

    beta_counts = BetaConfusionCounts()
    beta_mae_sum = 0.0
    beta_mae_n = 0

    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch_to_device(batch, torch_device)
        _, logs = policy.compute_loss(batch, class_weights=weights_on_device)
        for key in log_keys:
            sums[key] += float(logs[key])
        n_batches += 1

        raw_policy = _unwrap_policy(policy)
        beta_logit = raw_policy.beta_logits_from_batch(batch)
        beta_counts += beta_confusion_counts_from_logits(
            beta_logit,
            batch["beta_label"],
            threshold=0.5,
            high_prob_threshold=0.9,
        )
        mae_sum, mae_n = beta_mae_from_logits(beta_logit, batch["beta_label"])
        beta_mae_sum += mae_sum
        beta_mae_n += mae_n

    sums, n_batches = _ddp_reduce_epoch_metrics(sums, n_batches)
    metrics = _aggregate_logs(log_keys, sums, n_batches)
    metrics.update(beta_metrics_from_counts(beta_counts, high_prob_threshold=0.9))
    metrics["beta_mae"] = beta_mae_sum / beta_mae_n if beta_mae_n > 0 else float("nan")
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


def _config_snapshot(cfg: Any) -> Any:
    """Serialize Hydra cfg for checkpoint storage."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    if isinstance(cfg, dict):
        return cfg
    return cfg


def save_checkpoint(
    path: Path,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Any,
    *,
    optimizer_beta: torch.optim.Optimizer | None = None,
    ema_policy: Any | None = None,
    ema: Any | None = None,
    lr_scheduler: Any | None = None,
    lr_scheduler_beta: Any | None = None,
    lr_schedule: dict[str, Any] | None = None,
    train_state: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    from dataclasses import asdict

    from soda.training.checkpoint_utils import capture_rng_state, resume_training_payload

    policy = _unwrap_policy(policy)  # strip DDP wrapper before saving
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "policy": policy.state_dict(),
        "ema_policy": ema_policy.state_dict() if ema_policy is not None else None,
        "optimizer": optimizer.state_dict(),
        "optimizer_beta": optimizer_beta.state_dict() if optimizer_beta is not None else None,
        "normalizer": policy.normalizer.state_dict(),
        "low_policy_config": asdict(policy.cfg),
        "horizon": int(policy.horizon),
        "cfg": _config_snapshot(cfg),
        "metrics": metrics or {},
        **resume_training_payload(
            lr_scheduler=lr_scheduler,
            lr_scheduler_beta=lr_scheduler_beta,
            lr_schedule=lr_schedule,
            ema=ema,
            train_state=train_state,
            rng_state=capture_rng_state(),
        ),
    }
    torch.save(payload, path)


def _resolve_output_dir(cfg: Any, train_cfg: TrainLowConfig) -> Path:
    from soda.experiments.paths import infer_task_slug, train_low_dir

    if train_cfg.output_dir:
        return Path(train_cfg.output_dir)
    task = infer_task_slug(cfg)
    name = str(_cfg_get(cfg, "name", "soda_train_low"))
    return Path(train_low_dir(task, name))


def _run_epoch_loop(
    *,
    policy: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_backbone: torch.optim.Optimizer,
    optimizer_beta: torch.optim.Optimizer,
    lr_scheduler: Any,
    lr_scheduler_beta: Any,
    plateau_scheduler_backbone: Any | None,
    plateau_scheduler_beta: Any | None,
    train_batch_sampler: Any | None,
    val_batch_sampler: Any | None,
    ema_model: Any | None,
    ema: Any | None,
    class_weights: torch.Tensor | None,
    cfg: Any,
    device: str,
    num_epochs: int,
    warmup_epochs: int,
    mask_diffusion_on_positive: bool,
    skip_diffusion: bool,
    best_checkpoint_metric: str,
    checkpoint_every: int,
    out_dir: "Path",
    lr_schedule_meta: dict,
    wandb_run: Any | None,
    history: list[dict],
    best_val: float = float("inf"),
    checkpoint_prefix: str = "",
    best_checkpoint_name: str = "best.ckpt",
    ddp_rank: int = 0,
    epoch_offset: int = 0,
) -> tuple[list[dict], float]:
    """Run the training epoch loop; returns (history, best_val_score)."""
    from soda.training.checkpoint_utils import write_metrics_history

    def _ckpt_kwargs(metrics: dict[str, float]) -> dict[str, Any]:
        train_state: dict[str, Any] = {
            "best_val_score": best_val,
            "best_checkpoint_metric": best_checkpoint_metric,
        }
        if plateau_scheduler_backbone is not None:
            train_state["plateau_scheduler_backbone"] = plateau_scheduler_backbone.state_dict()
        if plateau_scheduler_beta is not None:
            train_state["plateau_scheduler_beta"] = plateau_scheduler_beta.state_dict()
        return {
            "optimizer_beta": optimizer_beta,
            "ema_policy": ema_model,
            "ema": ema,
            "lr_scheduler": lr_scheduler,
            "lr_scheduler_beta": lr_scheduler_beta,
            "lr_schedule": lr_schedule_meta,
            "train_state": train_state,
            "metrics": metrics,
        }

    for epoch in range(1, num_epochs + 1):
        global_epoch = epoch + epoch_offset
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(global_epoch)
        if val_batch_sampler is not None:
            val_batch_sampler.set_epoch(0)
        train_metrics = train_one_epoch(
            policy,
            train_loader,
            optimizer_backbone,
            optimizer_beta,
            device,
            ema=ema,
            class_weights=class_weights,
            mask_diffusion_on_positive=mask_diffusion_on_positive,
            skip_diffusion=skip_diffusion,
        )
        current_lr = optimizer_backbone.param_groups[0]["lr"]
        current_lr_beta = optimizer_beta.param_groups[0]["lr"]
        eval_policy = ema_model if ema_model is not None else policy
        val_metrics = validate(eval_policy, val_loader, device, class_weights=class_weights)

        if plateau_scheduler_backbone is not None:
            if epoch <= warmup_epochs:
                lr_scheduler.step()
                lr_scheduler_beta.step()
            else:
                plateau_scheduler_backbone.step(val_metrics["loss_diffusion"])
                plateau_scheduler_beta.step(val_metrics["loss_termination"])
        else:
            lr_scheduler.step()
            lr_scheduler_beta.step()

        row = {
            "epoch": global_epoch,
            "lr": current_lr,
            "lr_beta": current_lr_beta,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        if ddp_rank == 0:
            print(json.dumps(row))
            write_metrics_history(out_dir / "metrics.json", history)

            if wandb_run is not None:
                import wandb

                wandb.log(row, step=global_epoch)

        score = _checkpoint_score(val_metrics, best_checkpoint_metric)
        if score < best_val:
            best_val = score
            if ddp_rank == 0:
                save_checkpoint(
                    out_dir / best_checkpoint_name,
                    policy,
                    optimizer_backbone,
                    global_epoch,
                    cfg,
                    **_ckpt_kwargs(val_metrics),
                )

        ckpt_name = f"epoch_{global_epoch:04d}.ckpt"
        if (epoch % checkpoint_every == 0 or epoch == num_epochs) and ddp_rank == 0:
            save_checkpoint(
                out_dir / ckpt_name,
                policy,
                optimizer_backbone,
                global_epoch,
                cfg,
                **_ckpt_kwargs(val_metrics),
            )

    return history, best_val


def run_training(cfg: Any) -> None:
    from omegaconf import OmegaConf

    from soda.experiments.run_readme import (
        begin_training_run_archive,
        finalize_training_run_archive,
    )

    train_cfg = TrainLowConfig.from_hydra(cfg)
    _seed_all(train_cfg.seed)

    ddp_rank, ddp_world_size = 0, 1
    if _ddp_is_active():
        ddp_rank, ddp_world_size, local_device = _ddp_init(train_cfg.device)
        train_cfg.device = local_device

    out_dir = _resolve_output_dir(cfg, train_cfg)
    if ddp_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    _ddp_barrier()

    run_archive = None
    if ddp_rank == 0:
        run_archive = begin_training_run_archive(
            out_dir,
            kind="train_low",
            hydra_readme=train_cfg.run_readme,
            extra_metadata={
                "config_name": str(_cfg_get(cfg, "name", out_dir.name)),
                "num_epochs": train_cfg.num_epochs,
            },
        )

    train_ds, val_ds = build_datasets(cfg, all_anchors=train_cfg.all_anchors)
    resolved_pos_weight = _resolve_termination_pos_weight(
        cfg, train_ds, configured=train_cfg.termination_pos_weight
    )
    if train_cfg.termination_pos_weight is None:
        train_cfg.termination_pos_weight = resolved_pos_weight

    beta_batch_pos_fraction = None
    if train_cfg.beta_stratified_batches:
        from soda.dataset.option_stratified_sampler import positives_per_batch
        from soda.training.option_balance import resolve_beta_batch_pos_fraction

        beta_batch_pos_fraction = resolve_beta_batch_pos_fraction(
            None,
            termination_pos_weight=resolved_pos_weight,
        )
        n_pos = positives_per_batch(train_cfg.batch_size, beta_batch_pos_fraction)
        print(
            "Beta stratified batches (train + val): "
            f"pos_fraction={beta_batch_pos_fraction:.4f} "
            f"({n_pos}/{train_cfg.batch_size} beta=1 per batch)"
        )

    train_loader, val_loader, train_batch_sampler, val_batch_sampler = build_dataloaders(
        cfg,
        train_ds,
        val_ds,
        train_cfg,
        beta_batch_pos_fraction=beta_batch_pos_fraction,
        all_anchors=train_cfg.all_anchors,
    )
    policy, optimizer_backbone, optimizer_beta = build_policy_and_optimizer(
        cfg,
        train_ds,
        train_cfg,
        termination_loss_weight=train_cfg.termination_loss_weight,
        termination_pos_weight=train_cfg.termination_pos_weight,
    )
    class_weights = _resolve_option_class_weights(cfg, train_ds)

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

    # Load SODA warmstart checkpoint if specified via train_low.checkpoint.
    if train_cfg.checkpoint:
        _reload_phase_checkpoint(
            _unwrap_policy(policy),
            ema_model,
            train_cfg.checkpoint,
            train_cfg.device,
        )

    warmup_epochs = min(train_cfg.lr_warmup_epochs, train_cfg.num_epochs)
    plateau_scheduler_backbone = None
    plateau_scheduler_beta = None
    cosine_epochs = 0

    if train_cfg.lr_schedule_type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer_backbone, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
        )
        lr_scheduler_beta = torch.optim.lr_scheduler.LinearLR(
            optimizer_beta, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
        )
        _plateau_base = {"mode": "min", "factor": train_cfg.lr_plateau_factor, "min_lr": train_cfg.lr_plateau_min_lr}
        beta_patience = train_cfg.lr_plateau_patience_beta if train_cfg.lr_plateau_patience_beta is not None else train_cfg.lr_plateau_patience
        plateau_scheduler_backbone = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_backbone, patience=train_cfg.lr_plateau_patience, **_plateau_base
        )
        plateau_scheduler_beta = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_beta, patience=beta_patience, **_plateau_base
        )
    elif train_cfg.lr_schedule_type == "constant":
        # Constant LR with optional warmup. Uses LambdaLR to avoid the SequentialLR
        # bug where sub-scheduler __init__ corrupts the optimizer LR.
        wu = warmup_epochs
        def _const_warmup_fn(ep: int) -> float:
            if wu <= 0 or ep >= wu:
                return 1.0
            return max(1e-6, ep / wu)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_backbone, lr_lambda=_const_warmup_fn)
        lr_scheduler_beta = torch.optim.lr_scheduler.LambdaLR(optimizer_beta, lr_lambda=_const_warmup_fn)
    else:
        # Cosine: linear warmup then cosine anneal to 0.
        cosine_epochs = max(1, train_cfg.num_epochs - warmup_epochs)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer_backbone,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer_backbone, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_backbone, T_max=cosine_epochs, eta_min=0.0
                ),
            ],
            milestones=[warmup_epochs],
        )
        lr_scheduler_beta = torch.optim.lr_scheduler.SequentialLR(
            optimizer_beta,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer_beta, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer_beta, T_max=cosine_epochs, eta_min=0.0
                ),
            ],
            milestones=[warmup_epochs],
        )

    from soda.training.checkpoint_utils import build_lr_schedule_meta

    lr_schedule_meta = build_lr_schedule_meta(
        train_cfg,
        warmup_epochs=warmup_epochs,
        cosine_epochs=cosine_epochs,
    )

    lr_sched_desc = train_cfg.lr_schedule_type
    if train_cfg.lr_schedule_type == "plateau":
        _beta_pat = train_cfg.lr_plateau_patience_beta if train_cfg.lr_plateau_patience_beta is not None else train_cfg.lr_plateau_patience
        lr_sched_desc = (
            f"plateau(backbone->loss_diffusion patience={train_cfg.lr_plateau_patience},"
            f"b->loss_termination patience={_beta_pat},"
            f"factor={train_cfg.lr_plateau_factor},"
            f"min_lr={train_cfg.lr_plateau_min_lr})"
        )
    if ddp_rank == 0:
        raw = _unwrap_policy(policy)
        print(
            f"π_low train: {len(train_ds)} segments (train), "
            f"{len(val_ds)} segments (val), "
            f"horizon={train_ds.horizon}, "
            f"num_options={_resolve_num_options(cfg, train_ds)}, "
            f"option_balance={train_cfg.option_balance}, "
            f"use_ema={train_cfg.use_ema}, "
            f"lr_warmup_epochs={warmup_epochs}, "
            f"lr_schedule={lr_sched_desc}, "
            f"best_checkpoint_metric={train_cfg.best_checkpoint_metric}, "
            f"termination_input={raw.cfg.termination_input}, "
            f"termination_stop_grad={raw.cfg.termination_stop_grad}"
        )

    wandb_run = None
    if train_cfg.wandb_enabled and ddp_rank == 0:
        wandb_run = _init_wandb_run(train_cfg, cfg, out_dir)

    history, best_val = _run_epoch_loop(
        policy=policy,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_backbone=optimizer_backbone,
        optimizer_beta=optimizer_beta,
        lr_scheduler=lr_scheduler,
        lr_scheduler_beta=lr_scheduler_beta,
        plateau_scheduler_backbone=plateau_scheduler_backbone,
        plateau_scheduler_beta=plateau_scheduler_beta,
        train_batch_sampler=train_batch_sampler,
        val_batch_sampler=val_batch_sampler,
        ema_model=ema_model,
        ema=ema,
        class_weights=class_weights,
        cfg=cfg,
        device=train_cfg.device,
        num_epochs=train_cfg.num_epochs,
        warmup_epochs=warmup_epochs,
        mask_diffusion_on_positive=train_cfg.mask_diffusion_on_positive,
        skip_diffusion=train_cfg.skip_diffusion_loss,
        best_checkpoint_metric=train_cfg.best_checkpoint_metric,
        checkpoint_every=train_cfg.checkpoint_every,
        out_dir=out_dir,
        lr_schedule_meta=lr_schedule_meta,
        wandb_run=wandb_run,
        history=[],
        best_val=float("inf"),
        ddp_rank=ddp_rank,
    )

    if ddp_rank == 0:
        train_state: dict[str, Any] = {
            "best_val_score": best_val,
            "best_checkpoint_metric": train_cfg.best_checkpoint_metric,
        }
        if plateau_scheduler_backbone is not None:
            train_state["plateau_scheduler_backbone"] = plateau_scheduler_backbone.state_dict()
        if plateau_scheduler_beta is not None:
            train_state["plateau_scheduler_beta"] = plateau_scheduler_beta.state_dict()
        save_checkpoint(
            out_dir / "latest.ckpt",
            policy,
            optimizer_backbone,
            train_cfg.num_epochs,
            cfg,
            optimizer_beta=optimizer_beta,
            ema_policy=ema_model,
            ema=ema,
            lr_scheduler=lr_scheduler,
            lr_scheduler_beta=lr_scheduler_beta,
            lr_schedule=lr_schedule_meta,
            train_state=train_state,
            metrics=history[-1] if history else {},
        )

        if wandb_run is not None:
            import wandb
            wandb.finish()

        if run_archive is not None:
            finalize_training_run_archive(run_archive)

    _ddp_cleanup()


def main(cfg: Any) -> None:
    """Hydra entrypoint."""
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
