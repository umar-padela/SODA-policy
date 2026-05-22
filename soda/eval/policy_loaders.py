"""
Load policies for Push-T eval (DP frozen baseline today; SODA when trained).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
DP_ROOT = REPO_ROOT / "third_party" / "diffusion_policy"


class PushTPolicy(Protocol):
    """Minimal interface required by ``pusht_rollout`` (matches DP BaseImagePolicy)."""

    device: Any
    dtype: Any

    def reset(self) -> None: ...
    def predict_action(self, obs_dict: dict) -> dict: ...


@dataclass
class DPEvalSettings:
    """Env / horizon settings stored in a DP checkpoint."""

    n_obs_steps: int
    # Actions in each predict_action chunk (DP horizon; not an eval knob).
    policy_horizon: int
    # DP training default execute count (informational; eval uses runner n_action_steps).
    checkpoint_n_action_steps: int
    max_steps: int
    legacy_test: bool
    past_action_visible: bool

    @property
    def n_action_steps(self) -> int:
        """Backward-compatible alias for checkpoint_n_action_steps."""
        return self.checkpoint_n_action_steps


# SODA uses the same env fields as DP for Push-T sim eval.
SodaEvalSettings = DPEvalSettings


def _require_diffusion_policy() -> Path:
    if not (DP_ROOT / "eval.py").is_file():
        raise FileNotFoundError(
            f"Missing {DP_ROOT}. Run: git submodule update --init third_party/diffusion_policy"
        )
    from soda.eval.gym_vector_compat import apply_gym_dp_vector_compat

    apply_gym_dp_vector_compat()
    return DP_ROOT


def _settings_from_cfg(cfg: Any) -> DPEvalSettings:
    task = cfg.task
    horizon = int(getattr(cfg, "horizon", cfg.n_obs_steps + cfg.n_action_steps))
    return DPEvalSettings(
        n_obs_steps=int(cfg.n_obs_steps),
        policy_horizon=horizon,
        checkpoint_n_action_steps=int(cfg.n_action_steps),
        max_steps=int(task.env_runner.max_steps),
        legacy_test=bool(task.env_runner.legacy_test),
        past_action_visible=bool(getattr(cfg, "past_action_visible", False)),
    )


def load_dp_image_policy_and_cfg(
    checkpoint: str | Path,
    device: str = "cuda:0",
) -> tuple[Any, DPEvalSettings, Any]:
    """
    Load frozen Columbia DP Push-T *image* policy and Hydra cfg from ``.ckpt``.
    """
    import sys

    import dill
    import hydra
    import torch

    _require_diffusion_policy()
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(DP_ROOT)],
        check=True,
        cwd=str(DP_ROOT),
    )

    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    checkpoint = Path(checkpoint)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(checkpoint.parent / "_eval_load_tmp"))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(torch.device(device))
    policy.eval()

    return policy, _settings_from_cfg(cfg), cfg


def load_dp_image_policy(
    checkpoint: str | Path,
    device: str = "cuda:0",
) -> tuple[Any, DPEvalSettings]:
    """
    Load frozen Columbia DP Push-T *image* policy from ``.ckpt`` payload.
    """
    policy, settings, _cfg = load_dp_image_policy_and_cfg(checkpoint, device=device)
    return policy, settings


def load_frozen_dp_obs_encoder(
    checkpoint: str | Path,
    device: str = "cuda:0",
) -> tuple[Any, Any, int, int]:
    """
    Extract frozen hybrid DP vision stack for π_high ``ObsEncoder``.

    Returns
    -------
    obs_encoder
        ``dp_policy.obs_encoder`` (robomimic ResNet path on Push-T).
    normalizer
        ``dp_policy.normalizer`` — must be applied before ``obs_encoder``.
    n_obs_steps
        Observation history length (typically 2).
    global_feat_dim
        ``obs_feature_dim * n_obs_steps`` (DP ``global_cond`` size).
    """
    dp_policy, _settings, _cfg = load_dp_image_policy_and_cfg(checkpoint, device=device)

    obs_encoder = dp_policy.obs_encoder
    normalizer = dp_policy.normalizer
    n_obs_steps = int(dp_policy.n_obs_steps)
    global_feat_dim = int(dp_policy.obs_feature_dim) * n_obs_steps

    obs_encoder.eval()
    for param in obs_encoder.parameters():
        param.requires_grad = False

    return obs_encoder, normalizer, n_obs_steps, global_feat_dim


def load_soda_eval_cfg(config_name: str = "soda_supervised") -> Any:
    from omegaconf import OmegaConf

    path = REPO_ROOT / "configs" / "pusht" / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"SODA config not found: {path}. Expected configs/pusht/{config_name}.yaml"
        )
    return OmegaConf.load(path)


def load_soda_policy_and_cfg(
    checkpoint: str | Path,
    device: str = "cuda:0",
    *,
    config_name: str = "soda_supervised",
) -> tuple[Any, SodaEvalSettings, Any]:
    """
    Load SODA hierarchical policy + eval yaml (parallel to ``load_dp_image_policy_and_cfg``).

    Weights loading is not implemented yet; returns ``HierarchicalPolicy`` scaffold.
    """
    import torch

    from soda.inference.hierarchical_controller import (
        HierarchicalPolicy,
        HierarchicalPolicyConfig,
    )

    checkpoint = Path(checkpoint)
    cfg = load_soda_eval_cfg(config_name)
    settings = _settings_from_cfg(cfg)

    option_key = str(cfg.task.dataset.get("option_id_key", "option_id_supervised"))
    policy = HierarchicalPolicy(
        device=torch.device(device),
        config=HierarchicalPolicyConfig(
            config_name=config_name,
            option_id_key=option_key,
            n_obs_steps=int(cfg.n_obs_steps),
            checkpoint_n_action_steps=int(cfg.n_action_steps),
        ),
        checkpoint=checkpoint,
    )
    # TODO: torch.load SODA checkpoint; attach pi_high / pi_low; set policy.dtype
    _ = checkpoint  # reserved for weight load
    return policy, settings, cfg


def load_soda_policy(
    checkpoint: str | Path,
    device: str = "cuda:0",
    *,
    config_name: str = "soda_supervised",
) -> tuple[Any, SodaEvalSettings]:
    """Load trained SODA hierarchical policy."""
    policy, settings, _cfg = load_soda_policy_and_cfg(
        checkpoint, device=device, config_name=config_name
    )
    return policy, settings
