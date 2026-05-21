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
    # DP training default execute count (informational; eval uses ControlParams).
    checkpoint_n_action_steps: int
    max_steps: int
    legacy_test: bool
    past_action_visible: bool

    @property
    def n_action_steps(self) -> int:
        """Backward-compatible alias for checkpoint_n_action_steps."""
        return self.checkpoint_n_action_steps


def _require_diffusion_policy() -> Path:
    if not (DP_ROOT / "eval.py").is_file():
        raise FileNotFoundError(
            f"Missing {DP_ROOT}. Run: git submodule update --init third_party/diffusion_policy"
        )
    return DP_ROOT


def load_dp_image_policy(
    checkpoint: str | Path,
    device: str = "cuda:0",
) -> tuple[Any, DPEvalSettings]:
    """
    Load frozen Columbia DP Push-T *image* policy from ``.ckpt`` payload.
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

    task = cfg.task
    horizon = int(getattr(cfg, "horizon", cfg.n_obs_steps + cfg.n_action_steps))
    settings = DPEvalSettings(
        n_obs_steps=int(cfg.n_obs_steps),
        policy_horizon=horizon,
        checkpoint_n_action_steps=int(cfg.n_action_steps),
        max_steps=int(task.env_runner.max_steps),
        legacy_test=bool(task.env_runner.legacy_test),
        past_action_visible=bool(getattr(cfg, "past_action_visible", False)),
    )
    return policy, settings


def load_soda_policy(
    checkpoint: str | Path,
    device: str = "cuda:0",
    *,
    config_name: str = "soda_supervised",
) -> tuple[Any, DPEvalSettings]:
    """Load trained SODA hierarchical policy (§7 row 17 — not implemented yet)."""
    raise NotImplementedError(
        "SODA checkpoint loading is not implemented yet. "
        "Train pi_low/pi_high first, then wire load_soda_policy() and "
        "soda.inference.hierarchical_controller."
    )
