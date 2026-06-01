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
    """Default ``experiments/{task}/train_dp/{name}/`` unless ``train_dp.output_dir`` is set."""
    from soda.experiments.paths import infer_task_slug, train_dp_dir

    name = str(_cfg_get(yaml_cfg, "name", "dp"))
    train = _get_block(yaml_cfg, "train_dp")
    raw = _cfg_get(train, "output_dir", None)
    if raw:
        return Path(str(raw))
    task = infer_task_slug(yaml_cfg)
    return REPO_ROOT / train_dp_dir(task, name)


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

    # shape_meta for pusht_image (the only task we support via train_dp)
    shape_meta = {
        "obs": {
            "image": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [2]},
    }

    # noise_scheduler must live inside policy for Columbia workspace instantiation
    noise_scheduler.setdefault(
        "_target_", "diffusers.schedulers.scheduling_ddpm.DDPMScheduler"
    )

    policy.setdefault(
        "_target_",
        "diffusion_policy.policy.diffusion_unet_hybrid_image_policy."
        "DiffusionUnetHybridImagePolicy",
    )
    policy["horizon"] = horizon
    policy["n_obs_steps"] = n_obs_steps
    policy["n_action_steps"] = train_n_action_steps
    policy.setdefault("obs_as_global_cond", bool(_cfg_get(yaml_cfg, "obs_as_global_cond", True)))
    policy["shape_meta"] = shape_meta
    policy["noise_scheduler"] = noise_scheduler

    # Inject _target_ and horizon into dataset so Hydra can instantiate it.
    # num_workers is a DataLoader param, not a PushTImageDataset param — keep it
    # separately for the dataloader config but remove from dataset instantiation.
    dataset_cfg = dict(task.get("dataset", {}))
    num_workers = int(dataset_cfg.pop("num_workers", 8))
    dataset_cfg.setdefault(
        "_target_", "diffusion_policy.dataset.pusht_image_dataset.PushTImageDataset"
    )
    dataset_cfg["horizon"] = horizon
    task["dataset"] = dataset_cfg

    # Inject _target_ and required params into env_runner
    er = dict(task.get("env_runner", {}))
    er.setdefault(
        "_target_", "diffusion_policy.env_runner.pusht_image_runner.PushTImageRunner"
    )
    er.setdefault("n_train", 6)
    er.setdefault("n_train_vis", 2)
    er.setdefault("train_start_seed", 0)
    er.setdefault("n_test", 50)
    er.setdefault("n_test_vis", 4)
    er.setdefault("legacy_test", True)
    er.setdefault("test_start_seed", 100000)
    er.setdefault("max_steps", 300)
    er.setdefault("n_envs", None)
    er["n_action_steps"] = train_n_action_steps
    er["n_obs_steps"] = n_obs_steps
    er["past_action"] = bool(_cfg_get(yaml_cfg, "past_action_visible", False))
    task["env_runner"] = er

    workspace = {
        "_target_": _cfg_get(
            _get_block(yaml_cfg, "train_dp"),
            "_target_",
            "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace."
            "TrainDiffusionUnetHybridWorkspace",
        ),
        "name": name,
        "shape_meta": shape_meta,
        "horizon": horizon,
        "n_obs_steps": n_obs_steps,
        "n_action_steps": train_n_action_steps,
        "past_action_visible": bool(_cfg_get(yaml_cfg, "past_action_visible", False)),
        "obs_as_global_cond": bool(_cfg_get(yaml_cfg, "obs_as_global_cond", True)),
        "task": task,
        "policy": policy,
        "ema": {
            "_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel",
            "update_after_step": 0,
            "inv_gamma": 1.0,
            "power": 0.75,
            "min_value": 0.0,
            "max_value": 0.9999,
        },
        "checkpoint": {
            "topk": {
                "monitor_key": "test_mean_score",
                "mode": "max",
                "k": 5,
                "format_str": "epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt",
            },
            "save_last_ckpt": True,
            "save_last_snapshot": False,
        },
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
            "lr_scheduler": str(_cfg_get(train, "lr_scheduler", "cosine")),
            "lr_warmup_steps": int(_cfg_get(train, "lr_warmup_steps", 500)),
            "resume": bool(_cfg_get(train, "resume", True)),
            "debug": bool(_cfg_get(train, "debug", False)),
            "gradient_accumulate_every": int(_cfg_get(train, "gradient_accumulate_every", 1)),
            "sample_every": int(_cfg_get(train, "sample_every", 5)),
            "max_train_steps": _cfg_get(train, "max_train_steps", None),
            "max_val_steps": _cfg_get(train, "max_val_steps", None),
            "tqdm_interval_sec": float(_cfg_get(train, "tqdm_interval_sec", 1.0)),
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
            "num_workers": num_workers,
            "shuffle": True,
            "pin_memory": True,
            "persistent_workers": False,
        },
        "val_dataloader": {
            "batch_size": int(_cfg_get(train, "batch_size", 64)),
            "num_workers": num_workers,
            "shuffle": False,
            "pin_memory": True,
            "persistent_workers": False,
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
