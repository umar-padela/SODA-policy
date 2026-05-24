"""
Train π_high with flow matching on option segment starts (project_plan §7 row 15).

Pairs with ``soda/models/high_policy.py`` (``HighPolicy.compute_loss``).

Training loop pattern: ``assignment1/hw1/main.py::train_flow_matching_policy``.

Entrypoints
-----------
- Local: ``python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised``
- Modal: ``modal run modal/modal_train_high.py``

Hydra
-----
- ``high_policy:`` and ``train_high:`` in ``configs/pusht/soda_*.yaml``
- Reuses ``task.dataset.zarr_path``, ``option_id_key`` from same zarr as π_low
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from soda.dataset.option_start_dataset import (
    OptionStartDataset,
    build_option_start_dataset_from_config,
)
from soda.models.high_policy import HighPolicy, HighPolicyConfig

__all__ = [
    "HighPolicy",
    "HighPolicyConfig",
    "OptionStartDataset",
    "TrainHighConfig",
    "build_datasets",
    "build_dataloaders",
    "build_policy_and_optimizer",
    "train_one_epoch",
    "validate",
    "save_checkpoint",
    "run_training",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainHighConfig:
    """Training loop settings (mirror ``train_low.TrainLowConfig``)."""

    device: str = "cuda:0"
    seed: int = 42
    num_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-6
    checkpoint_every: int = 10
    output_dir: str | None = None
    wandb_enabled: bool = False
    wandb_project: str = "soda-train-high"
    low_checkpoint: str | None = None
    dp_checkpoint: str | None = None
    num_workers: int = 4
    option_balance: str = "none"
    use_ema: bool = False
    lr_warmup_epochs: int = 5
    run_readme: str | None = None

    @classmethod
    def from_hydra(cls, cfg: Any) -> TrainHighConfig:
        block = _get_block(cfg, "train_high")
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
            wandb_project=str(_cfg_get(block, "wandb_project", "soda-train-high")),
            low_checkpoint=_cfg_get(block, "low_checkpoint", None),
            dp_checkpoint=_cfg_get(block, "dp_checkpoint", None),
            num_workers=int(_cfg_get(block, "num_workers", 4)),
            option_balance=str(_cfg_get(block, "option_balance", "none")),
            use_ema=bool(_cfg_get(block, "use_ema", False)),
            lr_warmup_epochs=int(_cfg_get(block, "lr_warmup_epochs", 5)),
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


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_hydra_cfg(cfg: Any, *, train: bool) -> Any:
    """Build cfg slice for ``build_option_start_dataset_from_config``."""
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
                    "train": train,
                },
            )
        }
    )
    return merged


def build_datasets(cfg: Any) -> tuple[OptionStartDataset, OptionStartDataset]:
    train_ds = build_option_start_dataset_from_config(
        _dataset_hydra_cfg(cfg, train=True)
    )
    val_ds = train_ds.get_validation_dataset()
    return train_ds, val_ds


def build_dataloaders(
    cfg: Any,
    train_ds: OptionStartDataset,
    val_ds: OptionStartDataset,
    train_cfg: TrainHighConfig,
) -> tuple[DataLoader, DataLoader]:
    ds_cfg = _cfg_get(_cfg_get(cfg, "task", {}), "dataset", {})
    num_workers = int(_cfg_get(ds_cfg, "num_workers", train_cfg.num_workers))
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=str(train_cfg.device).startswith("cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(train_cfg.device).startswith("cuda"),
        drop_last=False,
    )
    return train_loader, val_loader


def _infer_num_options(dataset: OptionStartDataset) -> int:
    ids = np.asarray([seg.option_id for seg in dataset.segments])
    return int(len(np.unique(ids)))


def _resolve_num_options(cfg: Any, dataset: OptionStartDataset) -> int:
    configured = _cfg_get(_get_block(cfg, "high_policy"), "num_options", None)
    if configured is not None:
        return int(configured)
    return _infer_num_options(dataset)


def _resolve_low_checkpoint(train_cfg: TrainHighConfig) -> Path:
    if not train_cfg.low_checkpoint:
        raise FileNotFoundError(
            "train_high.low_checkpoint is not set. Point it at a π_low "
            "train_low checkpoint (e.g. experiments/train_low/soda_supervised/best.ckpt)."
        )
    path = Path(train_cfg.low_checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"low_checkpoint not found: {path}")
    return path


def _resolve_dp_checkpoint(train_cfg: TrainHighConfig) -> Path:
    if train_cfg.dp_checkpoint:
        path = Path(train_cfg.dp_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"dp_checkpoint not found: {path}")
        return path

    candidates = [
        REPO_ROOT / "experiments" / "pusht" / "dp_frozen" / "latest.ckpt",
        Path("/experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt"),
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path

    raise FileNotFoundError(
        "Frozen vision checkpoint not found. Set train_high.low_checkpoint (trained π_low) "
        "or train_high.dp_checkpoint (Columbia DP), or download DP via "
        "scripts/download_dp_pusht_checkpoint.sh (local) or modal download_frozen_dp."
    )


def _high_policy_config(
    cfg: Any,
    dataset: OptionStartDataset,
    *,
    global_feat_dim: int,
) -> HighPolicyConfig:
    block = _get_block(cfg, "high_policy")
    configured_gfd = _cfg_get(block, "global_feat_dim", None)
    if configured_gfd is not None and int(configured_gfd) != global_feat_dim:
        raise ValueError(
            f"high_policy.global_feat_dim={configured_gfd} != "
            f"encoder global_feat_dim={global_feat_dim}"
        )
    return HighPolicyConfig(
        num_options=_resolve_num_options(cfg, dataset),
        option_embed_dim=int(_cfg_get(block, "option_embed_dim", 32)),
        global_feat_dim=global_feat_dim,
        fm_hidden_dim=int(_cfg_get(block, "fm_hidden_dim", 256)),
        fm_num_layers=int(_cfg_get(block, "fm_num_layers", 3)),
        time_embed_dim=int(_cfg_get(block, "time_embed_dim", 32)),
        num_inference_steps=int(_cfg_get(block, "num_inference_steps", 10)),
    )


def build_policy_and_optimizer(
    cfg: Any,
    train_dataset: OptionStartDataset,
    train_cfg: TrainHighConfig,
) -> tuple[HighPolicy, torch.optim.Optimizer]:
    from soda.models.high_policy import ObsEncoder

    if train_cfg.low_checkpoint:
        low_ckpt = _resolve_low_checkpoint(train_cfg)
        obs_encoder = ObsEncoder.from_low_policy_checkpoint(
            low_ckpt, device=train_cfg.device, eval_cfg=cfg, freeze=True
        )
        global_feat_dim = obs_encoder.global_feat_dim
        hp_cfg = _high_policy_config(
            cfg, train_dataset, global_feat_dim=global_feat_dim
        )
        policy = HighPolicy(hp_cfg, obs_encoder).to(train_cfg.device)
    else:
        from soda.eval.policy_loaders import load_frozen_dp_obs_encoder

        dp_ckpt = _resolve_dp_checkpoint(train_cfg)
        _, _, _, global_feat_dim = load_frozen_dp_obs_encoder(
            dp_ckpt, device="cpu"
        )
        hp_cfg = _high_policy_config(
            cfg, train_dataset, global_feat_dim=global_feat_dim
        )
        policy = HighPolicy.from_dp_checkpoint(
            hp_cfg, dp_ckpt, device=train_cfg.device
        )
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


def _resolve_option_class_weights(
    cfg: Any,
    train_dataset: OptionStartDataset,
) -> torch.Tensor | None:
    from soda.training.option_balance import (
        count_option_ids,
        option_ids_from_segments,
        resolve_option_class_weights,
    )

    num_options = _resolve_num_options(cfg, train_dataset)
    mode = str(_cfg_get(_get_block(cfg, "train_high"), "option_balance", "none"))
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


def _batch_option_loss(
    policy: HighPolicy,
    batch: dict[str, Any],
    *,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss, logs = policy.compute_loss(batch, class_weights=class_weights)
    return loss, {"loss_fm": logs["loss"]}


def train_one_epoch(
    policy: HighPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    *,
    ema: Any | None = None,
    class_weights: torch.Tensor | None = None,
) -> dict[str, float]:
    policy.train()
    torch_device = torch.device(device)
    log_keys = ("loss_fm",)
    sums = {k: 0.0 for k in log_keys}
    n_batches = 0
    weights_on_device = (
        class_weights.to(torch_device) if class_weights is not None else None
    )

    for batch in tqdm(loader, desc="train", leave=False):
        batch = _move_batch_to_device(batch, torch_device)
        loss, logs = _batch_option_loss(
            policy, batch, class_weights=weights_on_device
        )
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
    policy: HighPolicy,
    loader: DataLoader,
    device: str,
) -> dict[str, float]:
    policy.eval()
    torch_device = torch.device(device)
    log_keys = ("loss_fm", "option_acc")
    sums = {k: 0.0 for k in log_keys}
    n_batches = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="val", leave=False):
        batch = _move_batch_to_device(batch, torch_device)
        _, logs = _batch_option_loss(policy, batch)
        sums["loss_fm"] += float(logs["loss_fm"])

        global_feat = policy.encode_obs(batch["obs"])
        pred = policy.sample_option(global_feat)
        option_id = batch["option_id"].reshape(-1).long()
        correct += int((pred == option_id).sum().item())
        total += int(option_id.numel())
        n_batches += 1

    metrics = _aggregate_logs(["loss_fm"], sums, n_batches)
    metrics["option_acc"] = float(correct / total) if total > 0 else float("nan")
    return metrics


def save_checkpoint(
    path: Path,
    policy: HighPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: Any,
    *,
    ema_policy: Any | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    normalizer = policy.obs_encoder.normalizer
    payload = {
        "epoch": epoch,
        "policy": policy.state_dict(),
        "ema_policy": ema_policy.state_dict() if ema_policy is not None else None,
        "optimizer": optimizer.state_dict(),
        "normalizer": normalizer.state_dict() if hasattr(normalizer, "state_dict") else None,
        "high_policy_config": asdict(policy.cfg),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def _resolve_output_dir(cfg: Any, train_cfg: TrainHighConfig) -> Path:
    if train_cfg.output_dir:
        return Path(train_cfg.output_dir)
    name = str(_cfg_get(cfg, "name", "soda_train_high"))
    return Path("experiments") / "train_high" / name


def run_training(cfg: Any) -> None:
    from omegaconf import OmegaConf

    from soda.experiments.run_readme import (
        begin_training_run_archive,
        finalize_training_run_archive,
    )

    train_cfg = TrainHighConfig.from_hydra(cfg)
    _seed_all(train_cfg.seed)
    out_dir = _resolve_output_dir(cfg, train_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_archive = begin_training_run_archive(
        out_dir,
        kind="train_high",
        hydra_readme=train_cfg.run_readme,
        extra_metadata={
            "config_name": str(_cfg_get(cfg, "name", out_dir.name)),
            "num_epochs": train_cfg.num_epochs,
        },
    )

    train_ds, val_ds = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_ds, val_ds, train_cfg)
    policy, optimizer = build_policy_and_optimizer(cfg, train_ds, train_cfg)
    class_weights = _resolve_option_class_weights(cfg, train_ds)

    # EMA — same config as train_low (inv_gamma=1, power=0.75, max=0.9999)
    ema_model = None
    ema = None
    if train_cfg.use_ema:
        from soda.eval.policy_loaders import _ensure_diffusion_policy_path
        _ensure_diffusion_policy_path()
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

    # LR scheduler: linear warmup → cosine decay (per-epoch)
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
        f"π_high train: {len(train_ds)} segment starts (train), "
        f"{len(val_ds)} segment starts (val), "
        f"num_options={_resolve_num_options(cfg, train_ds)}, "
        f"option_balance={train_cfg.option_balance}, "
        f"use_ema={train_cfg.use_ema}, "
        f"lr_warmup_epochs={warmup_epochs}, "
        f"vision={'π_low' if train_cfg.low_checkpoint else 'frozen DP'}"
    )

    wandb_run = None
    if train_cfg.wandb_enabled:
        import wandb

        wandb_run = wandb.init(
            project=train_cfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=str(out_dir),
        )

    best_val_acc = float("-inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, train_cfg.num_epochs + 1):
        train_metrics = train_one_epoch(
            policy,
            train_loader,
            optimizer,
            train_cfg.device,
            ema=ema,
            class_weights=class_weights,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        lr_scheduler.step()
        eval_policy = ema_model if ema_model is not None else policy
        val_metrics = validate(eval_policy, val_loader, train_cfg.device)
        row = {
            "epoch": epoch,
            "lr": current_lr,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))

        if wandb_run is not None:
            import wandb

            wandb.log(row, step=epoch)

        if val_metrics["option_acc"] > best_val_acc:
            best_val_acc = val_metrics["option_acc"]
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
