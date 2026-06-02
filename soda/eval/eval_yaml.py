"""
Load Push-T eval settings from a yaml config file (``dp_frozen.yaml``, ``soda_supervised.yaml``, …).

Modal eval passes ``--config configs/pusht/dp_frozen.yaml``; CLI flags override yaml fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from soda.eval.dp_frozen import DEFAULT_FROZEN_DP_VOLUME_PATH
from soda.eval.metrics import DEFAULT_OVERLAP_CHECKPOINTS
from soda.eval.policy_loaders import (
    _cfg_get,
    _yaml_attr,
    load_pusht_yaml,
    resolve_soda_checkpoint_paths,
)

PolicySource = Literal["dp_frozen", "dp", "soda", "soda_low"]
TaskName = Literal["pusht", "square"]

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvalCliOverrides:
    """Optional Modal / local CLI values that override yaml ``eval:*``."""

    checkpoint_path: Path | None = None
    high_checkpoint: Path | None = None
    low_checkpoint: Path | None = None
    full: bool = False
    n_action_steps: int | None = None
    n_test: int | None = None
    max_steps: int | None = None
    n_test_vis: int | None = None
    record_video: bool | None = None
    ckpt_slug: str | None = None
    policy: PolicySource | None = None
    fixed_option_id: int | None = None
    run_readme: str | None = None
    noise_eta: float = 0.0
    noise_rho: float = 0.9
    noise_scale_by_magnitude: bool = False


@dataclass
class EvalConfig:
    """Resolved eval run configuration."""

    config_path: Path
    config_name: str
    checkpoint_path: Path | None = None
    task: TaskName = "pusht"
    policy_source: PolicySource = "dp_frozen"
    ckpt_slug: str | None = None
    device: str = "cuda:0"
    n_action_steps: int = 8
    max_steps: int = 300
    overlap_checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS
    n_test: int = 5
    n_train: int = 0
    test_start_seed: int = 100000
    train_start_seed: int = 0
    show_progress: bool = True
    record_video: bool = True
    n_test_vis: int | None = None
    high_checkpoint: Path | None = None
    low_checkpoint: Path | None = None
    fixed_option_id: int | None = None
    wandb_enabled: bool = True
    wandb_project: str = "soda-eval"
    run_readme: str | None = None
    noise_eta: float = 0.0
    noise_rho: float = 0.9
    noise_scale_by_magnitude: bool = False

    @property
    def soda_config_name(self) -> str:
        """Backward-compatible alias used by run naming and SODA loaders."""
        return self.config_name


def resolve_eval_config_path(config: str | Path) -> Path:
    """Resolve yaml path relative to repo root when not absolute."""
    path = Path(config)
    if path.is_file():
        return path.resolve()
    candidate = REPO_ROOT / path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Eval config not found: {config}")


def _infer_policy_source(yaml_cfg: Any, *, config_path: Path) -> PolicySource:
    eval_block = _yaml_attr(yaml_cfg, "eval")
    explicit = _cfg_get(eval_block, "policy", None)
    if explicit is not None:
        policy = str(explicit).strip().lower()
        if policy == "dp":
            return "dp"
        if policy in ("dp_frozen", "dp_baseline", "diffusion_policy"):
            return "dp_frozen"
        if policy == "soda":
            return "soda"
        if policy in ("soda_low", "low_only", "low"):
            return "soda_low"
        raise ValueError(f"Unknown eval.policy: {explicit!r}")

    if _yaml_attr(yaml_cfg, "inference") is not None:
        return "soda"
    name = _cfg_get(yaml_cfg, "name", None)
    if name is not None and str(name).startswith("soda"):
        return "soda"
    stem = config_path.stem
    if stem == "dp":
        return "dp"
    if stem == "dp_frozen":
        return "dp_frozen"
    return "dp_frozen"


def _infer_task(yaml_cfg: Any) -> TaskName:
    task_block = _yaml_attr(yaml_cfg, "task")
    if isinstance(task_block, str):
        task = task_block
    elif task_block is not None:
        task = str(_cfg_get(task_block, "name", task_block))
    else:
        task = "pusht"
    if "square" in task.lower():
        return "square"
    return "pusht"


def _eval_regime(yaml_cfg: Any, *, full: bool) -> Any:
    eval_block = _yaml_attr(yaml_cfg, "eval")
    if eval_block is None:
        return {}
    regime = _cfg_get(eval_block, "full" if full else "smoke", None)
    return regime if regime is not None else {}


def build_eval_config_from_yaml(
    config: str | Path,
    *,
    cli: EvalCliOverrides | None = None,
    finalize_checkpoints: bool = True,
) -> EvalConfig:
    """
    Build :class:`EvalConfig` from a yaml file plus optional CLI overrides.

    Yaml layout (see ``configs/pusht/dp_frozen.yaml`` and ``soda_supervised.yaml``):

    - ``eval.policy`` — ``dp_frozen``, ``dp``, ``soda``, or ``soda_low``
    - ``n_action_steps`` (top-level) — replan window for sim eval
    - ``eval.smoke`` / ``eval.full`` — ``n_test`` (``max_steps`` from ``task.env_runner``)
    - DP: ``checkpoint.volume_path``
    - SODA: ``inference.high_checkpoint`` / ``inference.low_checkpoint``
    """
    cli = cli or EvalCliOverrides()
    config_path = resolve_eval_config_path(config)
    yaml_cfg = load_pusht_yaml(config_path)
    config_name = str(_cfg_get(yaml_cfg, "name", config_path.stem))
    policy_source = cli.policy if cli.policy is not None else _infer_policy_source(
        yaml_cfg, config_path=config_path
    )
    regime = _eval_regime(yaml_cfg, full=cli.full)
    eval_block = _yaml_attr(yaml_cfg, "eval")

    task_block = _yaml_attr(yaml_cfg, "task")
    env_runner = _cfg_get(task_block, "env_runner", None) if task_block is not None else None

    n_action_steps = int(
        cli.n_action_steps
        if cli.n_action_steps is not None
        else _cfg_get(yaml_cfg, "n_action_steps", 8)
    )
    n_test = int(
        cli.n_test
        if cli.n_test is not None
        else _cfg_get(regime, "n_test", 50 if cli.full else 5)
    )
    default_max_steps = 300
    if env_runner is not None:
        default_max_steps = int(_cfg_get(env_runner, "max_steps", default_max_steps))
    max_steps = int(
        cli.max_steps
        if cli.max_steps is not None
        else _cfg_get(regime, "max_steps", default_max_steps)
    )
    record_video = (
        cli.record_video
        if cli.record_video is not None
        else bool(_cfg_get(eval_block, "record_video", True))
    )
    n_test_vis = (
        cli.n_test_vis
        if cli.n_test_vis is not None
        else _cfg_get(eval_block, "n_test_vis", None)
    )

    test_start_seed = 100000
    if env_runner is not None:
        test_start_seed = int(_cfg_get(env_runner, "test_start_seed", test_start_seed))

    cfg = EvalConfig(
        config_path=config_path,
        config_name=config_name,
        task=_infer_task(yaml_cfg),
        policy_source=policy_source,
        ckpt_slug=cli.ckpt_slug,
        n_action_steps=n_action_steps,
        n_test=n_test,
        max_steps=max_steps,
        test_start_seed=test_start_seed,
        record_video=record_video,
        n_test_vis=int(n_test_vis) if n_test_vis is not None else None,
        checkpoint_path=cli.checkpoint_path,
        high_checkpoint=cli.high_checkpoint,
        low_checkpoint=cli.low_checkpoint,
        fixed_option_id=cli.fixed_option_id,
        wandb_enabled=bool(_cfg_get(eval_block, "wandb_enabled", True)),
        wandb_project=str(_cfg_get(eval_block, "wandb_project", "soda-eval")),
        run_readme=cli.run_readme or _cfg_get(eval_block, "run_readme", None),
        noise_eta=float(cli.noise_eta),
        noise_rho=float(cli.noise_rho),
    )
    if finalize_checkpoints:
        return finalize_eval_config(cfg, yaml_cfg=yaml_cfg)
    return cfg


def finalize_eval_config(
    config: EvalConfig,
    *,
    yaml_cfg: Any | None = None,
) -> EvalConfig:
    """Fill checkpoint paths from yaml (CLI paths already on ``config`` win)."""
    if yaml_cfg is None:
        yaml_cfg = load_pusht_yaml(config.config_path)

    if config.policy_source == "soda":
        infer_cfg = _yaml_attr(yaml_cfg, "inference")
        high_path, low_path = resolve_soda_checkpoint_paths(
            config.checkpoint_path,
            high_checkpoint=config.high_checkpoint,
            low_checkpoint=config.low_checkpoint,
            infer_cfg=infer_cfg,
        )
        config.high_checkpoint = high_path
        config.low_checkpoint = low_path
        config.checkpoint_path = high_path
    elif config.policy_source == "soda_low":
        if config.low_checkpoint is None and config.checkpoint_path is not None:
            config.low_checkpoint = Path(config.checkpoint_path)
        elif config.low_checkpoint is None:
            infer_cfg = _yaml_attr(yaml_cfg, "inference")
            low_path = _cfg_get(infer_cfg, "low_checkpoint", None) if infer_cfg else None
            if low_path is not None:
                config.low_checkpoint = Path(low_path)
        if config.low_checkpoint is None:
            raise ValueError(
                "soda_low eval requires a π_low checkpoint. "
                "Pass --checkpoint / --low-checkpoint or set inference.low_checkpoint in yaml."
            )
        config.checkpoint_path = config.low_checkpoint
        if config.fixed_option_id is None:
            raise ValueError(
                "soda_low eval requires --fixed-option-id (e.g. 0=REPOSITION, 1=LINEAR-PUSH, 2=PIVOT-PUSH)."
            )
    else:
        if config.checkpoint_path is None:
            ckpt_block = _yaml_attr(yaml_cfg, "checkpoint")
            volume_path = _cfg_get(ckpt_block, "volume_path", None)
            if volume_path:
                config.checkpoint_path = Path(volume_path)
            elif config.policy_source == "dp_frozen":
                config.checkpoint_path = DEFAULT_FROZEN_DP_VOLUME_PATH
    return config


def _resolve_checkpoint(config: EvalConfig) -> Path:
    if config.checkpoint_path is None:
        raise ValueError(
            f"No checkpoint in eval config {config.config_path}. "
            "Set checkpoint.volume_path (DP) or inference.* (SODA) in yaml, "
            "or pass --checkpoint / --high-checkpoint / --low-checkpoint."
        )
    path = Path(config.checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "For Columbia frozen DP on Modal, run "
            "`modal run modal/modal_download_dp.py` once."
        )
    return path
