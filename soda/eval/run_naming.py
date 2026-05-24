"""
Descriptive directory names for sim eval runs on the experiments volume.

Pattern (underscore-separated)::

    {task}_{policy}_h{H}_{smoke5|full50|n{N}}_ckpt-{slug}_t{steps}[_seed{S}]_utc{YYYYMMDD-HHMMSS}

``H`` is ``n_action_steps`` (8 = receding-horizon P0 default; 1 = closed-loop ablation).

Full checkpoint path is stored in ``eval_log.json``, not in the folder name.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

TaskName = Literal["pusht", "square"]
PolicyLabel = Literal["dp_frozen", "dp", "soda_supervised", "soda_unsupervised"]

DEFAULT_TEST_START_SEED = 10000
DEFAULT_MAX_STEPS = 300

_FROZEN_DP_DIR_MARKERS = ("dp_baselines", "pusht_image_cnn_train0")


def policy_label_from_config(
    policy_source: str,
    soda_config_name: str = "soda_supervised",
    *,
    fixed_option_id: int | None = None,
) -> PolicyLabel | str:
    if policy_source in ("dp_frozen", "dp_baseline"):
        return "dp_frozen"
    if policy_source == "dp":
        return "dp"
    if policy_source == "soda_low":
        if fixed_option_id is not None:
            return f"soda_low_o{int(fixed_option_id)}"
        return "soda_low"
    if policy_source == "soda":
        if soda_config_name == "soda_unsupervised":
            return "soda_unsupervised"
        return "soda_supervised"
    raise ValueError(f"Unknown policy_source: {policy_source!r}")


def _sanitize_token(value: str, *, max_len: int = 80) -> str:
    value = value.strip().lower()
    value = value.replace("=", "").replace(".", "_")
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "unknown"
    return value[:max_len]


def checkpoint_slug(checkpoint: Path, override: str | None = None) -> str:
    """
    Short checkpoint label for directory names.

    Override with ``ckpt_slug`` when the auto slug is ambiguous.
    """
    if override is not None:
        return _sanitize_token(override)

    path = Path(checkpoint)
    parts = path.parts
    stem = _sanitize_token(path.stem)

    if (
        stem == "latest"
        and any(marker in parts for marker in _FROZEN_DP_DIR_MARKERS)
    ):
        return "frozen_latest"

    parent = _sanitize_token(path.parent.name)
    if parent and parent not in {"checkpoints", "ckpt", "checkpoint", "checkpoints"}:
        return _sanitize_token(f"{parent}_{path.stem}")
    return stem


def _eval_size_token(n_test: int) -> str:
    if n_test == 5:
        return "smoke5"
    if n_test == 50:
        return "full50"
    return f"n{n_test}"


def build_eval_run_dir_name(
    *,
    task: TaskName,
    policy_label: PolicyLabel | str,
    n_action_steps: int,
    n_test: int,
    max_steps: int,
    checkpoint: Path,
    ckpt_slug: str | None = None,
    test_start_seed: int = DEFAULT_TEST_START_SEED,
    timestamp: datetime | None = None,
) -> str:
    """Return a single directory name (no slashes) for one eval run."""
    ts = timestamp or datetime.now(timezone.utc)
    utc = ts.strftime("%Y%m%d-%H%M%S")

    parts = [
        _sanitize_token(task),
        _sanitize_token(str(policy_label)),
        f"h{int(n_action_steps)}",
        _eval_size_token(int(n_test)),
        f"ckpt-{checkpoint_slug(checkpoint, ckpt_slug)}",
        f"t{int(max_steps)}",
    ]
    if int(test_start_seed) != DEFAULT_TEST_START_SEED:
        parts.append(f"seed{int(test_start_seed)}")
    parts.append(f"utc{utc}")
    return "_".join(parts)


