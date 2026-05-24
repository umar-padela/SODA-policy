"""Tests for SODA checkpoint path resolution (no GPU / full DP load)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soda.eval.policy_loaders import detect_checkpoint_kind, resolve_soda_checkpoint_paths


def test_detect_checkpoint_kind_high():
    assert detect_checkpoint_kind({"high_policy_config": {}, "policy": {}}) == "high"


def test_detect_checkpoint_kind_low():
    assert detect_checkpoint_kind({"policy": {}, "normalizer": {}}) == "low"


def test_detect_checkpoint_kind_combined():
    assert (
        detect_checkpoint_kind({"pi_high": {"policy": {}}, "pi_low": {"policy": {}}})
        == "combined"
    )


def test_detect_checkpoint_kind_unknown():
    with pytest.raises(ValueError, match="Unrecognized"):
        detect_checkpoint_kind({"epoch": 1})


def test_resolve_paths_from_explicit_kwargs(tmp_path: Path):
    high = tmp_path / "high.ckpt"
    low = tmp_path / "low.ckpt"
    high.write_bytes(b"x")
    low.write_bytes(b"y")
    torch.save({"high_policy_config": {}, "policy": {}}, high)
    torch.save({"policy": {}, "normalizer": {}}, low)

    h, l = resolve_soda_checkpoint_paths(
        high, high_checkpoint=high, low_checkpoint=low
    )
    assert h == high
    assert l == low


def test_resolve_from_infer_yaml_only_without_checkpoint_arg(tmp_path: Path):
    high = tmp_path / "high.ckpt"
    low = tmp_path / "low.ckpt"
    high.write_bytes(b"x")
    low.write_bytes(b"y")

    h, l = resolve_soda_checkpoint_paths(
        None,
        infer_cfg={"high_checkpoint": str(high), "low_checkpoint": str(low)},
    )
    assert h == high
    assert l == low


def test_resolve_yaml_only_raises_when_paths_missing():
    with pytest.raises(ValueError, match="SODA eval requires"):
        resolve_soda_checkpoint_paths(None, infer_cfg={"high_checkpoint": None})


def test_resolve_low_only_requires_high_in_infer(tmp_path: Path):
    low = tmp_path / "low.ckpt"
    torch.save({"policy": {}, "normalizer": {}}, low)

    with pytest.raises(ValueError, match="high_checkpoint"):
        resolve_soda_checkpoint_paths(low, infer_cfg={"low_checkpoint": None})

    high = tmp_path / "high.ckpt"
    high.write_bytes(b"x")
    h, l = resolve_soda_checkpoint_paths(
        low,
        infer_cfg={"high_checkpoint": str(high)},
    )
    assert l == low
    assert h == high
