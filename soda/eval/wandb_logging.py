"""Optional Weights & Biases logging for Push-T eval runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key, value in metrics.items():
        if key == "video_paths":
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[key] = value
    return out


def log_eval_to_wandb(
    result: dict[str, Any],
    *,
    enabled: bool,
    project: str,
    run_name: str | None = None,
) -> None:
    """Log eval scalars (and rollout videos when present) to W&B."""
    if not enabled:
        return

    import wandb

    metrics = dict(result.get("metrics") or {})
    config = dict(result.get("config") or {})
    name = (
        run_name
        or result.get("descriptive_run_name")
        or result.get("run_dir_name")
        or "eval"
    )

    wandb.init(
        project=project,
        name=str(name),
        job_type="eval",
        config=config,
    )
    wandb.log(_numeric_metrics(metrics))

    for path in metrics.get("video_paths") or []:
        video_path = Path(path)
        if video_path.is_file():
            wandb.log({f"video/{video_path.stem}": wandb.Video(str(video_path))})

    wandb.finish()
