"""
Build Columbia ``TrainDiffusionUnetHybridWorkspace`` Hydra config from ``configs/pusht/dp.yaml``.

Every key in ``configs/pusht/dp.yaml`` maps to a Columbia training or eval field.
Eval-only keys (``n_action_steps``, ``eval:*``, ``checkpoint.volume_path``) are excluded
from the workspace config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_block(cfg: Any, name: str) -> Any:
    return _cfg_get(cfg, name, {})


def _dataset_train_n_action_steps(yaml_cfg: Any) -> int:
    """Columbia dataloader execute window from ``task.dataset.pad_after + 1``."""
    ds = _cfg_get(_get_block(yaml_cfg, "task"), "dataset", {})
    pad_after = int(_cfg_get(ds, "pad_after", 7))
    return pad_after + 1


def resolve_dp_output_dir(yaml_cfg: Any) -> Path:
    """Default ``experiments/train_dp/{name}/`` unless ``train_dp.output_dir`` is set."""
    name = str(_cfg_get(yaml_cfg, "name", "dp"))
    train = _get_block(yaml_cfg, "train_dp")
    raw = _cfg_get(train, "output_dir", None)
    if raw:
        return Path(str(raw))
    return REPO_ROOT / "experiments" / "train_dp" / name


def build_columbia_workspace_cfg(yaml_cfg: Any) -> Any:
    """
    Convert repo ``dp.yaml`` → OmegaConf dict for Columbia workspace ``run()``.

    Requires ``third_party/diffusion_policy`` on ``PYTHONPATH``.
    """
    from omegaconf import OmegaConf

    name = str(_cfg_get(yaml_cfg, "name", "dp"))
    train = dict(_get_block(yaml_cfg, "train_dp") or {})
    train.pop("_target_", None)
    optimizer = dict(_get_block(yaml_cfg, "optimizer") or {})
    task = OmegaConf.to_container(_get_block(yaml_cfg, "task"), resolve=True)
    policy = dict(_get_block(yaml_cfg, "policy") or {})
    noise_scheduler = dict(_get_block(yaml_cfg, "noise_scheduler") or {})

    horizon = int(_cfg_get(yaml_cfg, "horizon", 16))
    n_obs_steps = int(_cfg_get(yaml_cfg, "n_obs_steps", 2))
    train_n_action_steps = _dataset_train_n_action_steps(yaml_cfg)
    task_name = str(_cfg_get(task, "name", "pusht_image"))
    wandb_enabled = bool(_cfg_get(train, "wandb_enabled", False))
    wandb_project = str(_cfg_get(train, "wandb_project", "soda-train-dp"))

    policy.setdefault(
        "_target_",
        "diffusion_policy.policy.diffusion_unet_hybrid_image_policy."
        "DiffusionUnetHybridImagePolicy",
    )
    policy["horizon"] = horizon
    policy["n_obs_steps"] = n_obs_steps
    policy["n_action_steps"] = train_n_action_steps
    policy.setdefault("obs_as_global_cond", bool(_cfg_get(yaml_cfg, "obs_as_global_cond", True)))

    workspace = {
        "_target_": _cfg_get(
            _get_block(yaml_cfg, "train_dp"),
            "_target_",
            "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace."
            "TrainDiffusionUnetHybridWorkspace",
        ),
        "name": name,
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": train_n_action_steps,
        "past_action_visible": bool(_cfg_get(yaml_cfg, "past_action_visible", False)),
        "obs_as_global_cond": bool(_cfg_get(yaml_cfg, "obs_as_global_cond", True)),
        "task": task,
        "policy": policy,
        "noise_scheduler": noise_scheduler,
        "training": {
            "seed": int(_cfg_get(train, "seed", 42)),
            "device": str(_cfg_get(train, "device", "cuda:0")),
            "num_epochs": int(_cfg_get(train, "num_epochs", 3050)),
            "batch_size": int(_cfg_get(train, "batch_size", 64)),
            "lr": float(_cfg_get(train, "lr", 1e-4)),
            "weight_decay": float(_cfg_get(train, "weight_decay", 1e-6)),
            "use_ema": bool(_cfg_get(train, "use_ema", True)),
            "checkpoint_every": int(_cfg_get(train, "checkpoint_every", 50)),
            "rollout_every": int(_cfg_get(train, "rollout_every", 50)),
            "val_every": int(_cfg_get(train, "val_every", 1)),
            "output_dir": str(resolve_dp_output_dir(yaml_cfg)),
        },
        "logging": {
            "group": None,
            "id": None,
            "mode": "online" if wandb_enabled else "disabled",
            "project": wandb_project,
            "name": f"{name}_{task_name}",
            "resume": True,
            "tags": [name, task_name],
        },
        "optimizer": optimizer,
        "dataloader": {
            "batch_size": int(_cfg_get(train, "batch_size", 64)),
            "num_workers": int(
                _cfg_get(_cfg_get(task, "dataset", {}), "num_workers", 8)
            ),
        },
        "val_dataloader": {
            "batch_size": int(_cfg_get(train, "batch_size", 64)),
            "num_workers": int(
                _cfg_get(_cfg_get(task, "dataset", {}), "num_workers", 8)
            ),
        },
    }
    return OmegaConf.create(workspace)


def merge_eval_yaml_into_ckpt_cfg(ckpt_cfg: Any, eval_yaml: Any) -> Any:
    """Overlay eval yaml env settings onto checkpoint cfg for sim runners."""
    from omegaconf import OmegaConf

    ckpt = OmegaConf.create(OmegaConf.to_container(ckpt_cfg, resolve=True))
    ev = OmegaConf.create(OmegaConf.to_container(eval_yaml, resolve=True))
    merged = OmegaConf.merge(ckpt, ev)
    return merged
