"""Tests for run README validation and training-run archives."""

from __future__ import annotations

from pathlib import Path

import pytest

from soda.experiments.run_readme import (
    RunReadmeError,
    begin_training_run_archive,
    copy_training_artifacts,
    finalize_training_run_archive,
    resolve_run_readme,
    validate_run_readme,
    write_run_readme,
)


def test_validate_run_readme_rejects_blank():
    with pytest.raises(RunReadmeError):
        validate_run_readme("   ")


def test_resolve_run_readme_prefers_env(monkeypatch):
    monkeypatch.setenv("SODA_RUN_README", "env description")
    assert resolve_run_readme(hydra_value="yaml description") == "env description"


def test_write_run_readme_creates_markdown(tmp_path: Path):
    write_run_readme(
        tmp_path,
        "Smoke test for pi_low",
        kind="train_low",
        metadata={"config_name": "soda_supervised"},
        artifacts=["best.ckpt"],
    )
    text = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Smoke test for pi_low" in text
    assert "best.ckpt" in text


def test_training_run_archive_copies_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("SODA_RUN_README", "Archive smoke")
    canonical = tmp_path / "soda_supervised"
    canonical.mkdir()
    (canonical / "best.ckpt").write_bytes(b"best")
    (canonical / "metrics.json").write_text("[]", encoding="utf-8")

    archive = begin_training_run_archive(
        canonical,
        kind="train_low",
        extra_metadata={"config_name": "soda_supervised"},
    )
    finalize_training_run_archive(archive)

    assert (archive.archive_dir / "README.md").is_file()
    assert (archive.archive_dir / "best.ckpt").read_bytes() == b"best"
    assert (archive.archive_dir / "metrics.json").is_file()


def test_copy_training_artifacts_includes_checkpoints_dir(tmp_path: Path):
    canonical = tmp_path / "dp"
    checkpoints = canonical / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "epoch=0001.ckpt").write_bytes(b"x")
    archive = tmp_path / "archive"
    copied = copy_training_artifacts(canonical, archive)
    assert "checkpoints/" in copied
    assert (archive / "checkpoints" / "epoch=0001.ckpt").is_file()
