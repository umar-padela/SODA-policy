"""
Run README notes and timestamped training-run archives.

Every train/eval entrypoint requires a non-empty description (Modal ``--run-readme``,
local ``SODA_RUN_README`` env, or Hydra ``train_*.run_readme`` / ``eval.run_readme``).

Training also copies checkpoints into ``{canonical_dir}/runs/{YYYYMMDD_HHMMSS}/`` after
each run while keeping the canonical ``best.ckpt`` / ``latest.ckpt`` layout unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RUN_README_ENV = "SODA_RUN_README"
INVOKE_COMMAND_ENV = "SODA_INVOKE_COMMAND"

TRAINING_ARTIFACT_PATTERNS = (
    "best.ckpt",
    "latest.ckpt",
    "metrics.json",
    "epoch_*.ckpt",
)


class RunReadmeError(ValueError):
    """Missing or invalid run description."""


def validate_run_readme(text: str | None) -> str:
    if text is None or not str(text).strip():
        raise RunReadmeError(
            "A non-empty run README is required. "
            "Pass --run-readme on ``modal run``, set SODA_RUN_README for local scripts, "
            "or set train_*.run_readme / eval.run_readme in Hydra overrides."
        )
    return str(text).strip()


def resolve_run_readme(*, hydra_value: str | None = None) -> str:
    """Resolve description from env var, then optional Hydra override."""
    env_value = os.environ.get(RUN_README_ENV)
    for candidate in (env_value, hydra_value):
        if candidate is not None and str(candidate).strip():
            return validate_run_readme(candidate)
    raise RunReadmeError(
        "A non-empty run README is required. "
        "Pass --run-readme on ``modal run``, set SODA_RUN_README for local scripts, "
        "or set train_*.run_readme / eval.run_readme in Hydra overrides."
    )


def utc_timestamp_label(timestamp: datetime | None = None) -> str:
    ts = timestamp or datetime.now(timezone.utc)
    return ts.strftime("%Y%m%d_%H%M%S")


def training_run_archive_dir(
    canonical_dir: Path,
    *,
    timestamp: datetime | None = None,
) -> Path:
    return Path(canonical_dir) / "runs" / utc_timestamp_label(timestamp)


def _invoke_command() -> str | None:
    raw = os.environ.get(INVOKE_COMMAND_ENV)
    return raw.strip() if raw and raw.strip() else None


def write_run_readme(
    output_dir: Path,
    description: str,
    *,
    kind: str,
    metadata: dict[str, Any] | None = None,
    artifacts: list[str] | None = None,
) -> Path:
    """Write ``README.md`` into ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = dict(metadata or {})
    meta.setdefault("kind", kind)
    meta.setdefault("description", validate_run_readme(description))

    lines = [
        "# Run notes",
        "",
        "## Description",
        "",
        meta["description"],
        "",
        "## Metadata",
        "",
    ]
    for key in sorted(meta):
        if key == "description":
            continue
        lines.append(f"- **{key}**: {meta[key]}")
    lines.append("")

    if artifacts:
        lines.extend(["## Artifacts", ""])
        lines.extend(f"- `{name}`" for name in artifacts)
        lines.append("")

    path = output_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    meta_path = output_dir / "run_readme.json"
    meta_path.write_text(
        json.dumps(
            {
                "description": meta["description"],
                "kind": kind,
                "metadata": meta,
                "artifacts": artifacts or [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def copy_training_artifacts(canonical_dir: Path, archive_dir: Path) -> list[str]:
    """Copy checkpoint artifacts from canonical dir into the timestamped archive."""
    canonical_dir = Path(canonical_dir)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for pattern in TRAINING_ARTIFACT_PATTERNS:
        for path in sorted(canonical_dir.glob(pattern)):
            dest = archive_dir / path.name
            shutil.copy2(path, dest)
            copied.append(path.name)

    checkpoints_dir = canonical_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        dest_checkpoints = archive_dir / "checkpoints"
        if dest_checkpoints.exists():
            shutil.rmtree(dest_checkpoints)
        shutil.copytree(checkpoints_dir, dest_checkpoints)
        copied.append("checkpoints/")

    return copied


@dataclass
class TrainingRunArchive:
    canonical_dir: Path
    archive_dir: Path
    readme: str
    kind: str
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


def begin_training_run_archive(
    canonical_dir: Path,
    *,
    kind: str,
    readme: str | None = None,
    hydra_readme: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> TrainingRunArchive:
    """Create ``runs/{timestamp}/`` and write the run README at train start."""
    canonical_dir = Path(canonical_dir)
    description = readme or resolve_run_readme(hydra_value=hydra_readme)
    archive_dir = training_run_archive_dir(canonical_dir)

    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dir": str(canonical_dir),
        "archive_dir": str(archive_dir),
        **(extra_metadata or {}),
    }
    invoke = _invoke_command()
    if invoke:
        metadata["invoke_command"] = invoke

    write_run_readme(
        archive_dir,
        description,
        kind=kind,
        metadata=metadata,
    )
    return TrainingRunArchive(
        canonical_dir=canonical_dir,
        archive_dir=archive_dir,
        readme=description,
        kind=kind,
        started_utc=metadata["started_utc"],
        metadata=metadata,
    )


def finalize_training_run_archive(archive: TrainingRunArchive) -> Path:
    """Copy training artifacts into the timestamped archive and refresh README."""
    artifacts = copy_training_artifacts(archive.canonical_dir, archive.archive_dir)
    metadata = {
        **archive.metadata,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dir": str(archive.canonical_dir),
        "archive_dir": str(archive.archive_dir),
    }
    write_run_readme(
        archive.archive_dir,
        archive.readme,
        kind=archive.kind,
        metadata=metadata,
        artifacts=artifacts,
    )
    print(f"Archived training run to {archive.archive_dir}")
    return archive.archive_dir
