"""
Eval run artifacts: config snapshot, command/overrides manifest, video renaming.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soda.eval.eval_yaml import EvalCliOverrides, EvalConfig


def config_stem_from_path(config_path: Path) -> str:
    """Folder name from yaml file (``dp_frozen``, ``soda_supervised``, …)."""
    return Path(config_path).stem


def build_eval_output_dir(
    experiments_root: Path,
    *,
    config_path: Path,
    task: str | None = None,
    timestamp: datetime | None = None,
) -> tuple[Path, datetime]:
    """
    ``{root}/{task}/eval/{config_stem}/{YYYYMMDD}/{HHMMSS}/``

    Returns ``(output_dir, timestamp)`` so callers reuse the same instant.
    """
    from soda.experiments.paths import infer_task_slug

    ts = timestamp or datetime.now(timezone.utc)
    stem = config_stem_from_path(config_path)
    task_slug = task or infer_task_slug(config_path=config_path)
    out = (
        Path(experiments_root)
        / task_slug
        / "eval"
        / stem
        / ts.strftime("%Y%m%d")
        / ts.strftime("%H%M%S")
    )
    return out, ts


def run_dir_name_from_path(output_dir: Path, experiments_root: Path) -> str:
    """Relative path under ``{task}/eval/`` for logs (e.g. ``pusht/dp_frozen/20260525/014851``)."""
    root = Path(experiments_root)
    for task in ("pusht", "square"):
        try:
            rel = output_dir.relative_to(root / task / "eval")
            return f"{task}/{rel.as_posix()}"
        except ValueError:
            continue
    try:
        rel = output_dir.relative_to(root / "eval")
        return rel.as_posix()
    except ValueError:
        return output_dir.name


def cli_overrides_dict(cli: EvalCliOverrides | None) -> dict[str, Any]:
    """Non-default CLI fields actually passed for this run."""
    if cli is None:
        return {}
    raw = asdict(cli)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if key == "full" and value is False:
            continue
        if key == "record_video" and value is True:
            continue
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def write_eval_run_manifest(
    output_dir: Path,
    *,
    config: EvalConfig,
    invoke_command: str | None = None,
    cli: EvalCliOverrides | None = None,
    timestamp: datetime | None = None,
    run_readme: str | None = None,
) -> None:
    """Snapshot yaml + record command, resolved eval settings, and run README."""
    from soda.experiments.run_readme import resolve_run_readme, write_run_readme

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    description = resolve_run_readme(
        hydra_value=run_readme or config.run_readme,
    )
    config.run_readme = description

    config_src = Path(config.config_path)
    if config_src.is_file():
        shutil.copy2(config_src, output_dir / "config.yaml")

    ts = timestamp or datetime.now(timezone.utc)
    manifest = {
        "timestamp_utc": ts.isoformat(),
        "config_path": str(config.config_path),
        "config_stem": config_stem_from_path(config.config_path),
        "invoke_command": invoke_command,
        "cli_overrides": cli_overrides_dict(cli),
        "resolved_eval": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
            if k != "overlap_checkpoints"
        },
        "resolved_eval_overlap_checkpoints": list(config.overlap_checkpoints),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if invoke_command:
        (output_dir / "command.txt").write_text(invoke_command + "\n", encoding="utf-8")

    write_run_readme(
        output_dir,
        description,
        kind="eval",
        metadata={
            "timestamp_utc": ts.isoformat(),
            "config_path": str(config.config_path),
            "config_stem": config_stem_from_path(config.config_path),
            "invoke_command": invoke_command,
            "output_dir": str(output_dir),
            "policy_source": config.policy_source,
        },
    )


def _score_token(score: float) -> str:
    """Overlap ratio → ``055`` for 55%."""
    return f"{int(round(float(score) * 100)):03d}"


def rename_eval_videos(
    output_dir: Path,
    *,
    config_stem: str,
    runner_log: dict[str, Any],
    prefix_filter: str = "test/",
) -> list[str]:
    """
    Rename ``media/*.mp4`` to ``{config_stem}_ep{idx:02d}_score{pct}.mp4``.

    Uses ``{prefix}sim_video_{seed}`` / ``{prefix}sim_max_reward_{seed}`` keys
    from the serialized runner log.
    """
    output_dir = Path(output_dir)
    media_dir = output_dir / "media"
    if not media_dir.is_dir():
        return []

    prefix_counts: dict[str, int] = {}
    renamed: list[str] = []
    video_re = re.compile(r"^(?P<prefix>.+)sim_video_(?P<seed>\d+)$")

    for key, path in sorted(runner_log.items()):
        if not isinstance(key, str) or path is None:
            continue
        match = video_re.match(key)
        if not match:
            continue

        prefix = match.group("prefix")
        if prefix_filter and prefix != prefix_filter:
            continue

        seed = match.group("seed")
        old_path = Path(str(path))
        if not old_path.is_file():
            candidate = media_dir / old_path.name
            if candidate.is_file():
                old_path = candidate
            else:
                continue

        reward_key = f"{prefix}sim_max_reward_{seed}"
        score = float(runner_log.get(reward_key, 0.0))
        ep_idx = prefix_counts.get(prefix, 0)
        prefix_counts[prefix] = ep_idx + 1

        new_name = (
            f"{config_stem}_ep{ep_idx:02d}_seed{seed}_score{_score_token(score)}.mp4"
        )
        new_path = media_dir / new_name
        if new_path.exists() and new_path.resolve() != old_path.resolve():
            stem = new_path.stem
            suffix = 1
            while new_path.exists():
                new_path = media_dir / f"{stem}_v{suffix}.mp4"
                suffix += 1

        if old_path.resolve() != new_path.resolve():
            shutil.move(str(old_path), str(new_path))

        runner_log[key] = str(new_path)
        renamed.append(str(new_path))

    return renamed
