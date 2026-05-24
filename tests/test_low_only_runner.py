"""Tests for π_low-only eval adapter (no GPU / sim)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soda.eval.eval_yaml import EvalCliOverrides, build_eval_config_from_yaml
from soda.eval.low_only_runner import FixedOptionLowPolicy
from soda.eval.run_naming import policy_label_from_config


class _StubLowPolicy:
    device = torch.device("cpu")
    dtype = torch.float32
    last_option_id: int | None = None

    def reset(self) -> None:
        self.last_option_id = None

    def predict_action(self, obs_dict, option_id):
        value = next(iter(obs_dict.values()))
        self.last_option_id = int(option_id.reshape(-1)[0].item())
        batch = int(value.shape[0])
        return {"action": torch.zeros(batch, 1, 2)}


def test_fixed_option_low_policy_passes_constant_option_id():
    stub = _StubLowPolicy()
    wrapped = FixedOptionLowPolicy(stub, fixed_option_id=2)
    obs = {"image": torch.zeros(1, 2, 3, 96, 96)}
    out = wrapped.predict_action(obs)
    assert stub.last_option_id == 2
    assert out["action"].shape == (1, 1, 2)


def test_policy_label_soda_low():
    assert policy_label_from_config("soda_low", fixed_option_id=0) == "soda_low_o0"


def test_build_eval_config_soda_low_requires_fixed_option(tmp_path: Path):
    cfg_path = tmp_path / "soda_supervised.yaml"
    low_ckpt = tmp_path / "best.ckpt"
    low_ckpt.write_bytes(b"x")
    cfg_path.write_text(
        "name: soda_supervised\ntask:\n  name: pusht_image\n  env_runner:\n    max_steps: 300\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed-option-id"):
        build_eval_config_from_yaml(
            cfg_path,
            cli=EvalCliOverrides(
                policy="soda_low",
                checkpoint_path=low_ckpt,
            ),
        )

    cfg = build_eval_config_from_yaml(
        cfg_path,
        cli=EvalCliOverrides(
            policy="soda_low",
            checkpoint_path=low_ckpt,
            fixed_option_id=1,
        ),
    )
    assert cfg.policy_source == "soda_low"
    assert cfg.fixed_option_id == 1
    assert cfg.low_checkpoint == low_ckpt
