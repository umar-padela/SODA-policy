"""
Train π_low for state-based (lowdim) tasks — block push.

  python soda/training/train_low_lowdim.py \\
      --config-path configs/block_push --config-name soda_lowdim

  modal run modal/modal_train_lowdim.py
"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from soda.dataset.option_aware_lowdim_dataset import OptionAwareLowdimDataset
from soda.models.low_policy_lowdim import LowdimSODAPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _require(cfg: Any, key: str) -> Any:
    v = _cfg(cfg, key)
    if v is None:
        raise ValueError(f"Config key '{key}' is required but not set.")
    return v


def _fit_normalizer(dataset: OptionAwareLowdimDataset):
    """Fit a LinearNormalizer on the full train split."""
    from diffusion_policy.model.common.normalizer import LinearNormalizer

    all_obs    = np.asarray(dataset._state[:])
    all_action = np.asarray(dataset._action[:])

    norm = LinearNormalizer()
    norm.fit({"obs": all_obs, "action": all_action})
    return norm


def _build_policy(cfg: Any, dataset: OptionAwareLowdimDataset, device: str) -> LowdimSODAPolicy:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    pcfg   = cfg.policy
    lcfg   = cfg.low_policy
    nscfg  = cfg.noise_scheduler

    obs_dim    = dataset._state.shape[1]
    action_dim = dataset._action.shape[1] + 1   # +1 for duration channel
    horizon    = dataset.horizon
    n_obs_steps    = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)
    num_options    = dataset.num_options
    option_embed_dim = int(_require(lcfg, "option_embed_dim"))

    noise_scheduler = DDPMScheduler(
        num_train_timesteps = int(_cfg(nscfg, "num_train_timesteps", 100)),
        beta_start          = float(_cfg(nscfg, "beta_start", 0.0001)),
        beta_end            = float(_cfg(nscfg, "beta_end", 0.02)),
        beta_schedule       = str(_cfg(nscfg, "beta_schedule", "squaredcos_cap_v2")),
        clip_sample         = bool(_cfg(nscfg, "clip_sample", True)),
        prediction_type     = str(_cfg(nscfg, "prediction_type", "epsilon")),
    )

    global_cond_dim = obs_dim * n_obs_steps   # widened by option_embed_dim inside LowdimSODAPolicy

    model = ConditionalUnet1D(
        input_dim              = action_dim,
        global_cond_dim        = global_cond_dim,
        diffusion_step_embed_dim = int(_cfg(pcfg, "diffusion_step_embed_dim", 128)),
        down_dims              = list(_cfg(pcfg, "down_dims", [256, 512, 1024])),
        kernel_size            = int(_cfg(pcfg, "kernel_size", 9)),
        n_groups               = int(_cfg(pcfg, "n_groups", 8)),
        cond_predict_scale     = bool(_cfg(pcfg, "cond_predict_scale", False)),
    )

    dp_policy_kwargs = dict(
        model              = model,
        noise_scheduler    = noise_scheduler,
        horizon            = horizon,
        obs_dim            = obs_dim,
        action_dim         = action_dim,
        n_action_steps     = n_action_steps,
        n_obs_steps        = n_obs_steps,
        num_inference_steps = int(_cfg(pcfg, "num_inference_steps", 100)),
        obs_as_global_cond = True,
    )

    policy = LowdimSODAPolicy(
        dp_policy_kwargs    = dp_policy_kwargs,
        num_options         = num_options,
        option_embed_dim    = option_embed_dim,
        termination_hidden_dim = int(_cfg(lcfg, "termination_hidden_dim", 256)),
    )

    normalizer = _fit_normalizer(dataset)
    policy.set_normalizer(normalizer)
    return policy.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: Any) -> None:
    tcfg   = cfg.train_low
    device = str(_require(tcfg, "device"))
    seed   = int(_cfg(tcfg, "seed", 42))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = _cfg(tcfg, "output_dir", None)
    if out_dir is None:
        out_dir = f"experiments/train_low/{cfg.name}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # ── Dataset ───────────────────────────────────────────────────────────────
    dcfg = cfg.task.dataset
    train_ds = OptionAwareLowdimDataset(
        zarr_path     = str(_require(dcfg, "zarr_path")),
        option_id_key = str(_cfg(dcfg, "option_id_key", "option_id_supervised")),
        horizon       = None,
        n_obs_steps   = int(cfg.n_obs_steps),
        min_segment_len = int(_cfg(dcfg, "min_segment_len", 3)),
        val_ratio     = float(_cfg(dcfg, "val_ratio", 0.1)),
        seed          = int(_cfg(dcfg, "seed", 42)),
        train         = True,
        all_anchors   = bool(_cfg(tcfg, "all_anchors", True)),
    )
    val_ds = train_ds.get_validation_dataset(all_anchors=True)

    print(f"Train segments: {len(train_ds)}  |  Val segments: {len(val_ds)}")
    print(f"num_options: {train_ds.num_options}  |  horizon: {train_ds.horizon}")

    n_workers = int(_cfg(tcfg, "num_workers", 4))
    train_loader = DataLoader(train_ds, batch_size=int(_cfg(tcfg, "batch_size", 256)),
                              shuffle=True, num_workers=n_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=int(_cfg(tcfg, "batch_size", 256)),
                              shuffle=False, num_workers=n_workers, pin_memory=True)

    # ── Policy ───────────────────────────────────────────────────────────────
    policy = _build_policy(cfg, train_ds, device)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # ── EMA ──────────────────────────────────────────────────────────────────
    use_ema = bool(_cfg(tcfg, "use_ema", True))
    ema_policy = copy.deepcopy(policy) if use_ema else None
    ema_decay  = 0.9999

    # ── Optimiser ────────────────────────────────────────────────────────────
    lr           = float(_require(tcfg, "lr"))
    weight_decay = float(_cfg(tcfg, "weight_decay", 1e-6))
    optimizer    = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_epochs  = int(_cfg(tcfg, "lr_warmup_epochs", 5))
    num_epochs     = int(_require(tcfg, "num_epochs"))
    total_steps    = num_epochs * len(train_loader)
    warmup_steps   = warmup_epochs * len(train_loader)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── W&B ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if bool(_cfg(tcfg, "wandb_enabled", False)):
        import wandb
        wandb_run = wandb.init(
            project = str(_cfg(tcfg, "wandb_project", "soda")),
            name    = _cfg(tcfg, "wandb_run_name", None),
            group   = _cfg(tcfg, "wandb_group", None),
            tags    = list(_cfg(tcfg, "wandb_tags", [])),
            config  = OmegaConf.to_container(cfg, resolve=True),
            dir     = str(out_dir),
        )

    # ── Training ─────────────────────────────────────────────────────────────
    term_loss_weight = float(_cfg(tcfg, "termination_loss_weight", 0.005))
    checkpoint_every = int(_cfg(tcfg, "checkpoint_every", 50))
    best_val_loss    = float("inf")
    global_step      = 0

    for epoch in range(1, num_epochs + 1):
        policy.train()
        epoch_sums = {"loss_diffusion": 0.0, "loss_termination": 0.0, "loss_total": 0.0}
        n_batches  = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)
        for batch in pbar:
            losses = policy.compute_loss(batch)
            loss   = losses["loss_diffusion"] + term_loss_weight * losses["loss_termination"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if use_ema:
                with torch.no_grad():
                    for p_ema, p in zip(ema_policy.parameters(), policy.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

            for k, v in losses.items():
                epoch_sums[k] += v.item()
            epoch_sums["loss_total"] += loss.item()
            n_batches += 1
            global_step += 1

        epoch_means = {k: v / n_batches for k, v in epoch_sums.items()}
        epoch_means["lr"] = scheduler.get_last_lr()[0]

        # ── Validation ───────────────────────────────────────────────────────
        eval_policy = ema_policy if use_ema else policy
        eval_policy.eval()
        val_sums = {"val_loss_diffusion": 0.0, "val_loss_total": 0.0}
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                vlosses = eval_policy.compute_loss(batch)
                val_sums["val_loss_diffusion"] += vlosses["loss_diffusion"].item()
                val_sums["val_loss_total"] += (
                    vlosses["loss_diffusion"] + term_loss_weight * vlosses["loss_termination"]
                ).item()
                val_n += 1
        val_means = {k: v / max(val_n, 1) for k, v in val_sums.items()}

        log = {**epoch_means, **val_means, "epoch": epoch}
        if wandb_run:
            wandb_run.log(log, step=global_step)

        if epoch % 10 == 0 or epoch == num_epochs:
            parts = "  ".join(f"{k}={v:.4f}" for k, v in log.items() if k != "epoch")
            print(f"[epoch {epoch:4d}]  {parts}")

        # ── Checkpointing ────────────────────────────────────────────────────
        if epoch % checkpoint_every == 0:
            torch.save(eval_policy.state_dict(), out_dir / f"checkpoint_ep{epoch:04d}.ckpt")

        val_loss = val_means["val_loss_diffusion"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(eval_policy.state_dict(), out_dir / "best.ckpt")

    torch.save(eval_policy.state_dict(), out_dir / "latest.ckpt")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Checkpoints at: {out_dir}")

    if wandb_run:
        wandb_run.finish()


# ---------------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------------

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: Any) -> None:
    # Add repo root to sys.path so imports work
    import sys
    repo_root = str(Path(__file__).parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    train(cfg)


if __name__ == "__main__":
    main()
