"""
Task-first experiment directory layout (local repo + Modal Volume).

Volume mount inside containers: ``/experiments``
CLI ``modal volume ls/get`` paths: relative to volume root (no ``/experiments`` prefix).

Layout::

    {local: experiments/ | volume: /experiments}/{task}/
        train_low/{config_name}/
        train_high/{config_name}/
        train_dp/{config_name}/
        sweep_low/{trial_tag}/
        sweep_high/{trial_tag}/
        segment_rollout/
        eval/{config_stem}/{YYYYMMDD}/{HHMMSS}/
        dp_baselines/{baseline_name}/     # Push-T frozen DP only today
        _hydra/{train_low|train_high|...}/{config_name}/   # Hydra scratch logs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

TaskSlug = Literal["pusht", "square"]

EXPERIMENTS_DIRNAME = "experiments"
VOLUME_MOUNT = "/experiments"
MODAL_VOLUME_NAME = "soda-experiments"

TRAIN_LOW = "train_low"
TRAIN_HIGH = "train_high"
TRAIN_DP = "train_dp"
SWEEP_LOW = "sweep_low"
SWEEP_HIGH = "sweep_high"
SEGMENT_ROLLOUT = "segment_rollout"
EVAL = "eval"
DP_BASELINES = "dp_baselines"
HYDRA = "_hydra"

FROZEN_DP_PUSHT_BASELINE = "pusht_image_cnn_train0"
FROZEN_DP_CKPT_NAME = "latest.ckpt"

_VALID_TASKS = frozenset({"pusht", "square"})


def normalize_task_slug(task: str) -> TaskSlug:
    slug = str(task).strip().lower()
    if slug in _VALID_TASKS:
        return slug  # type: ignore[return-value]
    if "square" in slug:
        return "square"
    return "pusht"


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def infer_task_slug(
    cfg: Any = None,
    *,
    config_path: str | Path | None = None,
) -> TaskSlug:
    """Infer ``pusht`` / ``square`` from config file path or Hydra cfg."""
    if config_path is not None:
        path = Path(config_path)
        parts = path.parts
        if "configs" in parts:
            idx = parts.index("configs")
            if idx + 1 < len(parts):
                candidate = parts[idx + 1].lower()
                if candidate in _VALID_TASKS:
                    return candidate  # type: ignore[return-value]
        if "square" in path.as_posix().lower():
            return "square"

    if cfg is not None:
        task_block = _cfg_get(cfg, "task", None)
        if isinstance(task_block, str):
            text = task_block
        elif task_block is not None:
            text = str(_cfg_get(task_block, "name", task_block))
        else:
            text = "pusht"
        if "square" in text.lower():
            return "square"

    return "pusht"


def local_task_root(task: str) -> Path:
    return Path(EXPERIMENTS_DIRNAME) / normalize_task_slug(task)


def volume_task_root(task: str) -> str:
    return f"{VOLUME_MOUNT}/{normalize_task_slug(task)}"


def _local_parts(task: str, *parts: str) -> Path:
    return local_task_root(task).joinpath(*parts)


def _volume_parts(task: str, *parts: str) -> str:
    return "/".join([volume_task_root(task), *parts])


def train_low_dir(task: str, config_name: str, *, in_volume: bool = False) -> Path | str:
    rel = (TRAIN_LOW, config_name)
    return _volume_parts(task, *rel) if in_volume else _local_parts(task, *rel)


def train_high_dir(task: str, config_name: str, *, in_volume: bool = False) -> Path | str:
    rel = (TRAIN_HIGH, config_name)
    return _volume_parts(task, *rel) if in_volume else _local_parts(task, *rel)


def train_dp_dir(task: str, config_name: str, *, in_volume: bool = False) -> Path | str:
    rel = (TRAIN_DP, config_name)
    return _volume_parts(task, *rel) if in_volume else _local_parts(task, *rel)


def sweep_low_dir(task: str, trial_tag: str, *, in_volume: bool = False) -> Path | str:
    rel = (SWEEP_LOW, trial_tag)
    return _volume_parts(task, *rel) if in_volume else _local_parts(task, *rel)


def sweep_high_dir(task: str, trial_tag: str, *, in_volume: bool = False) -> Path | str:
    rel = (SWEEP_HIGH, trial_tag)
    return _volume_parts(task, *rel) if in_volume else _local_parts(task, *rel)


def segment_rollout_dir(task: str, *, in_volume: bool = False) -> Path | str:
    rel = (SEGMENT_ROLLOUT,)
    return _volume_parts(task, *rel) if in_volume else _local_parts(task, *rel)


def eval_task_root(task: str, *, in_volume: bool = False) -> Path | str:
    rel = (EVAL,)
    return _volume_parts(task, *rel) if in_volume else _local_parts(task, *rel)


def frozen_dp_pusht_checkpoint(*, in_volume: bool = True) -> Path | str:
    # dp_baselines lives at the volume root, not under pusht/
    rel = (DP_BASELINES, FROZEN_DP_PUSHT_BASELINE, FROZEN_DP_CKPT_NAME)
    if in_volume:
        return f"{VOLUME_MOUNT}/" + "/".join(rel)
    return _local_parts("pusht", *rel)


def hydra_run_dir(
    task: str,
    hydra_subdir: str,
    config_name: str,
    *,
    in_volume: bool = False,
) -> Path | str:
    rel = (HYDRA, hydra_subdir, config_name)
    return _volume_parts(task, *rel) if in_volume else _local_parts(task, *rel)


def to_volume_experiments_path(path: str | Path) -> str:
    """Map ``experiments/...`` to ``/experiments/...`` for Modal Volume writes."""
    text = str(path).replace("\\", "/")
    if text.startswith(f"{VOLUME_MOUNT}/"):
        return text
    if text.startswith(f"{EXPERIMENTS_DIRNAME}/"):
        return f"{VOLUME_MOUNT}/{text[len(EXPERIMENTS_DIRNAME) + 1:]}"
    return text


def local_mirror_for_experiments_path(
    path: str | Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Map volume or repo-relative paths to ``{repo_root}/experiments/...``."""
    root = repo_root or Path(__file__).resolve().parents[2]
    text = str(path).replace("\\", "/")
    if text.startswith(f"{VOLUME_MOUNT}/"):
        return root / EXPERIMENTS_DIRNAME / text[len(f"{VOLUME_MOUNT}/") :]
    if Path(text).is_absolute():
        return Path(text)
    return root / text


def volume_relative_path(path: str | Path) -> str:
    """Strip ``/experiments/`` prefix for ``modal volume ls/get``."""
    text = str(path).replace("\\", "/")
    prefix = f"{VOLUME_MOUNT}/"
    if text.startswith(prefix):
        return text[len(prefix) :]
    if text.startswith(f"{EXPERIMENTS_DIRNAME}/"):
        return text[len(f"{EXPERIMENTS_DIRNAME}/") :]
    return text.lstrip("/")


def volume_metrics_remote_path(volume_output_dir: str | Path) -> str:
    """``metrics.json`` path for ``modal volume get`` (relative to volume root)."""
    return f"{volume_relative_path(volume_output_dir).rstrip('/')}/metrics.json"
