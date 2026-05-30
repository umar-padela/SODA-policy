"""
Load policies for Push-T eval (DP frozen baseline; trained SODA π_high + π_low).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
DP_ROOT = REPO_ROOT / "third_party" / "diffusion_policy"

CheckpointKind = Literal["high", "low", "combined"]


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
    policy_horizon: int
    checkpoint_n_action_steps: int
    max_steps: int
    legacy_test: bool
    past_action_visible: bool

    @property
    def n_action_steps(self) -> int:
        return self.checkpoint_n_action_steps


SodaEvalSettings = DPEvalSettings


def _ensure_diffusion_policy_path() -> Path:
    if not (DP_ROOT / "eval.py").is_file():
        raise FileNotFoundError(
            f"Missing {DP_ROOT}. Run: git submodule update --init third_party/diffusion_policy"
        )
    import sys

    dp_str = str(DP_ROOT)
    if dp_str not in sys.path:
        sys.path.insert(0, dp_str)
    return DP_ROOT


def _require_diffusion_policy() -> Path:
    _ensure_diffusion_policy_path()
    from soda.eval.gym_vector_compat import apply_gym_dp_vector_compat

    apply_gym_dp_vector_compat()
    return DP_ROOT


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _settings_from_cfg(cfg: Any, *, policy_horizon: int | None = None) -> DPEvalSettings:
    task = cfg.task
    yaml_horizon = _cfg_get(cfg, "horizon", None)
    if policy_horizon is not None:
        horizon = int(policy_horizon)
    elif yaml_horizon is not None:
        horizon = int(yaml_horizon)
    else:
        horizon = int(_cfg_get(cfg, "n_obs_steps", 2)) + int(
            _cfg_get(cfg, "n_action_steps", 8)
        )
    return DPEvalSettings(
        n_obs_steps=int(cfg.n_obs_steps),
        policy_horizon=horizon,
        checkpoint_n_action_steps=int(cfg.n_action_steps),
        max_steps=int(task.env_runner.max_steps),
        legacy_test=bool(task.env_runner.legacy_test),
        past_action_visible=bool(getattr(cfg, "past_action_visible", False)),
    )


def _load_torch_payload(checkpoint: Path) -> dict[str, Any]:
    import inspect

    import torch

    kwargs: dict[str, Any] = {"map_location": "cpu"}
    # PyTorch 2.0+; Modal image uses 1.12 — omit on older builds.
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(checkpoint, **kwargs)


def detect_checkpoint_kind(payload: dict[str, Any]) -> CheckpointKind:
    """Classify a SODA / DP training checkpoint payload."""
    if payload.get("pi_high") is not None and payload.get("pi_low") is not None:
        return "combined"
    if "high_policy_config" in payload:
        return "high"
    if "policy" in payload and "normalizer" in payload:
        return "low"
    raise ValueError(
        "Unrecognized checkpoint payload (expected π_high, π_low, or combined SODA ckpt)."
    )


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def resolve_soda_checkpoint_paths(
    checkpoint: str | Path | None = None,
    *,
    high_checkpoint: str | Path | None = None,
    low_checkpoint: str | Path | None = None,
    infer_cfg: Any | None = None,
) -> tuple[Path, Path]:
    """
    Resolve π_high and π_low checkpoint paths for hierarchical eval.

    Priority: explicit kwargs → ``inference.high_checkpoint`` / ``low_checkpoint``
    → auto-detect from ``checkpoint`` (requires the other path in ``inference``).

    When both paths are set (yaml and/or CLI), ``checkpoint`` may be omitted.
    """
    high_path = _resolve_path(high_checkpoint)
    low_path = _resolve_path(low_checkpoint)

    if infer_cfg is not None:
        if high_path is None:
            high_path = _resolve_path(_cfg_get(infer_cfg, "high_checkpoint", None))
        if low_path is None:
            low_path = _resolve_path(_cfg_get(infer_cfg, "low_checkpoint", None))

    if high_path is not None and low_path is not None:
        return high_path, low_path

    if checkpoint is None:
        raise ValueError(
            "SODA eval requires π_high and π_low checkpoints. "
            "Set inference.high_checkpoint and inference.low_checkpoint in "
            "configs/pusht/<config_name>.yaml, or pass --high-checkpoint / "
            "--low-checkpoint / --checkpoint on the CLI."
        )

    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    kind = detect_checkpoint_kind(_load_torch_payload(ckpt))
    if kind == "combined":
        return ckpt, ckpt

    if kind == "high":
        if high_path is None:
            high_path = ckpt
        if low_path is None:
            raise ValueError(
                f"π_high checkpoint at {ckpt}; set inference.low_checkpoint in yaml "
                "or pass low_checkpoint= to load_soda_policy_and_cfg."
            )
    elif kind == "low":
        if low_path is None:
            low_path = ckpt
        if high_path is None:
            raise ValueError(
                f"π_low checkpoint at {ckpt}; set inference.high_checkpoint in yaml "
                "or pass high_checkpoint= to load_soda_policy_and_cfg."
            )

    assert high_path is not None and low_path is not None
    return high_path, low_path


def load_dp_image_policy_and_cfg(
    checkpoint: str | Path,
    device: str = "cuda:0",
) -> tuple[Any, DPEvalSettings, Any]:
    """Load frozen Columbia DP Push-T *image* policy and Hydra cfg from ``.ckpt``."""
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
    policy, settings, _cfg = load_dp_image_policy_and_cfg(checkpoint, device=device)
    return policy, settings


def load_frozen_dp_obs_encoder(
    checkpoint: str | Path,
    device: str = "cuda:0",
) -> tuple[Any, Any, int, int]:
    """Extract frozen hybrid DP vision stack for π_high ``ObsEncoder``."""
    dp_policy, _settings, _cfg = load_dp_image_policy_and_cfg(checkpoint, device=device)

    obs_encoder = dp_policy.obs_encoder
    normalizer = dp_policy.normalizer
    n_obs_steps = int(dp_policy.n_obs_steps)
    global_feat_dim = int(dp_policy.obs_feature_dim) * n_obs_steps

    obs_encoder.eval()
    for param in obs_encoder.parameters():
        param.requires_grad = False

    return obs_encoder, normalizer, n_obs_steps, global_feat_dim


def load_imagenet_dp_obs_encoder(
    checkpoint: str | Path,
    device: str = "cpu",
) -> tuple[Any, Any, int, int]:
    """Build DP obs encoder with ImageNet weights (not Push-T fine-tuned).

    Instantiates the DP workspace from checkpoint config WITHOUT loading the
    model state dict, so the ResNet backbone retains ImageNet+GroupNorm weights.
    Only the normalizer (Push-T data statistics) is loaded from the checkpoint.
    """
    import sys

    import dill
    import torch

    _require_diffusion_policy()
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(DP_ROOT)],
        check=True,
        cwd=str(DP_ROOT),
    )

    import hydra
    from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: F401

    checkpoint = Path(checkpoint)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    # Build workspace from cfg only — NO load_payload → ResNet has ImageNet+GroupNorm weights
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=str(checkpoint.parent / "_imagenet_init_tmp"))
    policy = workspace.model

    # Load only the normalizer (Push-T mean/std) from checkpoint state dict
    model_state: dict[str, Any] = payload["state_dicts"]["model"]
    normalizer_state = {
        k[len("normalizer."):]: v
        for k, v in model_state.items()
        if k.startswith("normalizer.")
    }
    policy.normalizer.load_state_dict(normalizer_state)

    policy.to(torch.device(device))
    policy.eval()

    obs_encoder = policy.obs_encoder
    normalizer = policy.normalizer
    n_obs_steps = int(policy.n_obs_steps)
    global_feat_dim = int(policy.obs_feature_dim) * n_obs_steps

    obs_encoder.eval()
    for param in obs_encoder.parameters():
        param.requires_grad = False

    return obs_encoder, normalizer, n_obs_steps, global_feat_dim


def load_frozen_low_obs_encoder(
    checkpoint: str | Path,
    device: str = "cuda:0",
    *,
    eval_cfg: Any | None = None,
) -> tuple[Any, Any, int, int]:
    """Extract frozen hybrid vision stack from a trained π_low ``train_low`` checkpoint."""
    low_policy = load_low_policy_from_checkpoint(
        checkpoint, device=device, eval_cfg=eval_cfg
    )

    obs_encoder = low_policy.obs_encoder
    normalizer = low_policy.normalizer
    n_obs_steps = int(low_policy.n_obs_steps)
    global_feat_dim = int(low_policy.obs_feature_dim) * n_obs_steps

    obs_encoder.eval()
    for param in obs_encoder.parameters():
        param.requires_grad = False

    return obs_encoder, normalizer, n_obs_steps, global_feat_dim


def load_soda_eval_cfg(config_name: str = "soda_supervised") -> Any:
    """Load ``configs/pusht/{config_name}.yaml`` (training + eval)."""
    path = REPO_ROOT / "configs" / "pusht" / f"{config_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"SODA config not found: {path}. Expected configs/pusht/{config_name}.yaml"
        )
    return load_pusht_yaml(path)


def _yaml_attr(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a top-level yaml key from OmegaConf, dict, or namespace."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def load_pusht_yaml(path: str | Path) -> Any:
    """Load a Push-T yaml config from an absolute or repo-relative path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(path)
    except ImportError:
        pass
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Loading eval yaml requires omegaconf or PyYAML. "
            "Install project deps (conda env soda / Modal image)."
        ) from exc
    from types import SimpleNamespace

    def _to_ns(obj: Any) -> Any:
        if isinstance(obj, dict):
            return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_ns(v) for v in obj]
        return obj

    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return _to_ns(data) if isinstance(data, dict) else data


def load_high_policy_from_checkpoint(
    checkpoint: str | Path,
    device: str = "cuda:0",
    *,
    eval_cfg: Any | None = None,
) -> Any:
    """Load trained π_high from ``train_high`` checkpoint."""
    from omegaconf import OmegaConf

    from soda.models.high_policy import HighPolicy, HighPolicyConfig, ObsEncoder

    checkpoint = Path(checkpoint)
    payload = _load_torch_payload(checkpoint)
    if detect_checkpoint_kind(payload) == "combined":
        payload = payload["pi_high"]

    if "high_policy_config" not in payload:
        raise ValueError(f"Not a π_high checkpoint: {checkpoint}")

    saved_cfg = payload.get("cfg")
    hydra_cfg = OmegaConf.create(saved_cfg) if saved_cfg else eval_cfg
    if hydra_cfg is None:
        raise ValueError(
            f"Checkpoint {checkpoint} has no embedded cfg; pass eval_cfg or config_name."
        )

    hp_cfg = HighPolicyConfig(**payload["high_policy_config"])

    from soda.training.train_high import (
        TrainHighConfig,
        _resolve_dp_checkpoint,
        _resolve_low_checkpoint,
    )

    train_high = TrainHighConfig.from_hydra(hydra_cfg)
    if train_high.low_checkpoint:
        low_ckpt = _resolve_low_checkpoint(train_high)
        obs_encoder = ObsEncoder.from_low_policy_checkpoint(
            low_ckpt, device="cpu", eval_cfg=hydra_cfg
        )
    else:
        dp_ckpt = (
            Path(train_high.dp_checkpoint)
            if train_high.dp_checkpoint
            else _resolve_dp_checkpoint(train_high)
        )
        obs_encoder = ObsEncoder.from_dp_checkpoint(dp_ckpt, device="cpu")
    policy = HighPolicy(hp_cfg, obs_encoder)
    policy.load_state_dict(payload["policy"])
    if payload.get("normalizer") is not None and hasattr(
        policy.obs_encoder.normalizer, "load_state_dict"
    ):
        policy.obs_encoder.normalizer.load_state_dict(payload["normalizer"])

    torch_device = __import__("torch").device(device)
    policy.to(torch_device)
    policy.eval()
    return policy


def load_low_policy_from_checkpoint(
    checkpoint: str | Path,
    device: str = "cuda:0",
    *,
    eval_cfg: Any | None = None,
) -> Any:
    """Load trained π_low from ``train_low`` checkpoint."""
    from omegaconf import OmegaConf

    import torch

    _require_diffusion_policy()

    checkpoint = Path(checkpoint)
    payload = _load_torch_payload(checkpoint)
    if detect_checkpoint_kind(payload) == "combined":
        payload = payload["pi_low"]

    if "high_policy_config" in payload:
        raise ValueError(f"Expected π_low checkpoint, got π_high: {checkpoint}")

    saved_cfg = payload.get("cfg")
    hydra_cfg = OmegaConf.create(saved_cfg) if saved_cfg else eval_cfg
    if hydra_cfg is None:
        raise ValueError(
            f"Checkpoint {checkpoint} has no embedded cfg; pass eval_cfg or config_name."
        )

    from soda.models.low_policy import LowPolicyConfig
    from soda.training.train_low import (
        _build_low_policy,
        _dataset_hydra_cfg,
        build_low_policy_from_saved_config,
        build_option_dataset_from_config,
    )

    if payload.get("low_policy_config"):
        raw = dict(payload["low_policy_config"])
        th_raw = raw.pop("termination_head")
        raw.pop("horizon_stretch_max", None)
        raw.setdefault("termination_pos_weight", 1.0)
        raw.setdefault("termination_input", "bottleneck")
        raw.setdefault("termination_stop_grad", True)
        from soda.models.termination_head import TerminationHeadConfig

        lp_cfg = LowPolicyConfig(
            **raw,
            termination_head=TerminationHeadConfig(**th_raw),
        )
        saved_horizon = payload.get("horizon") or payload.get("horizon_stretch_max")
        if saved_horizon is None:
            dataset = build_option_dataset_from_config(
                _dataset_hydra_cfg(hydra_cfg, train=True, random_anchor=False)
            )
            saved_horizon = dataset.horizon
        policy = build_low_policy_from_saved_config(
            hydra_cfg, lp_cfg, horizon=int(saved_horizon)
        )
    else:
        dataset = build_option_dataset_from_config(
            _dataset_hydra_cfg(hydra_cfg, train=True, random_anchor=False)
        )
        policy = _build_low_policy(
            hydra_cfg,
            dataset,
            termination_pos_weight=1.0,
        )

    if payload.get("normalizer") is not None:
        policy.normalizer.load_state_dict(payload["normalizer"])
    # Prefer EMA weights when available (matches Columbia eval protocol)
    weights_key = "ema_policy" if payload.get("ema_policy") is not None else "policy"
    policy.load_state_dict(payload[weights_key])

    torch_device = torch.device(device)
    policy.to(torch_device)
    policy.eval()
    return policy


def load_low_policy_and_cfg(
    checkpoint: str | Path,
    device: str = "cuda:0",
    *,
    config_name: str = "soda_supervised",
) -> tuple[Any, SodaEvalSettings, Any]:
    """Load trained π_low plus yaml env settings (for low-only sim eval)."""
    eval_cfg = load_soda_eval_cfg(config_name)
    policy = load_low_policy_from_checkpoint(
        checkpoint, device=device, eval_cfg=eval_cfg
    )
    settings = _settings_from_cfg(eval_cfg, policy_horizon=int(policy.horizon))
    return policy, settings, eval_cfg


def load_soda_policy_and_cfg(
    checkpoint: str | Path | None = None,
    device: str = "cuda:0",
    *,
    config_name: str = "soda_supervised",
    high_checkpoint: str | Path | None = None,
    low_checkpoint: str | Path | None = None,
) -> tuple[Any, SodaEvalSettings, Any]:
    """
    Load ``HierarchicalPolicy`` with trained π_high + π_low weights attached.

    Provide both checkpoints via yaml ``inference.*``, CLI kwargs, or a single
    ``checkpoint`` path plus the sibling in yaml / kwargs.
    """
    import torch

    from soda.inference.hierarchical_controller import (
        HierarchicalPolicy,
        HierarchicalPolicyConfig,
    )

    eval_cfg = load_soda_eval_cfg(config_name)
    infer_cfg = getattr(eval_cfg, "inference", None)
    high_path, low_path = resolve_soda_checkpoint_paths(
        checkpoint,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
        infer_cfg=infer_cfg,
    )

    pi_high = load_high_policy_from_checkpoint(
        high_path, device=device, eval_cfg=eval_cfg
    )
    pi_low = load_low_policy_from_checkpoint(
        low_path, device=device, eval_cfg=eval_cfg
    )

    option_key = str(eval_cfg.task.dataset.get("option_id_key", "option_id_supervised"))
    beta_transition = float(
        _cfg_get(infer_cfg, "beta_transition", 0.5) if infer_cfg is not None else 0.5
    )
    beta_diffusion_t = int(
        _cfg_get(infer_cfg, "beta_diffusion_t", 0) if infer_cfg is not None else 0
    )

    torch_device = torch.device(device)
    policy = HierarchicalPolicy(
        device=torch_device,
        config=HierarchicalPolicyConfig(
            config_name=config_name,
            option_id_key=option_key,
            beta_transition=beta_transition,
            beta_diffusion_t=beta_diffusion_t,
        ),
        checkpoint=high_path,
        pi_high=pi_high,
        pi_low=pi_low,
    )
    policy.dtype = getattr(pi_low, "dtype", torch.float32)
    settings = _settings_from_cfg(eval_cfg, policy_horizon=int(pi_low.horizon))
    return policy, settings, eval_cfg


def load_soda_policy(
    checkpoint: str | Path,
    device: str = "cuda:0",
    *,
    config_name: str = "soda_supervised",
    high_checkpoint: str | Path | None = None,
    low_checkpoint: str | Path | None = None,
) -> tuple[Any, SodaEvalSettings]:
    policy, settings, _cfg = load_soda_policy_and_cfg(
        checkpoint,
        device=device,
        config_name=config_name,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
    )
    return policy, settings
