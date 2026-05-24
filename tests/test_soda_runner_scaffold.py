"""Scaffold tests for SODA eval pipeline (no GPU / sim)."""

from __future__ import annotations

from pathlib import Path

from soda.eval.policy_loaders import load_soda_eval_cfg
from soda.eval.runner_common import slice_action_chunk
from soda.eval.soda_runner import build_soda_pusht_runner
from soda.inference.hierarchical_controller import HierarchicalPolicy


def test_load_soda_eval_cfg_has_env_runner():
    cfg = load_soda_eval_cfg("soda_supervised")
    assert int(cfg.n_obs_steps) == 2
    assert cfg.task.env_runner.legacy_test is True
    assert cfg.task.dataset.option_id_key == "option_id_supervised"


def test_build_soda_pusht_runner_requires_diffusion_policy_submodule():
    cfg = load_soda_eval_cfg("soda_supervised")
    dp_root = Path(__file__).resolve().parents[1] / "third_party" / "diffusion_policy"
    if not (dp_root / "eval.py").is_file():
        return
    runner = build_soda_pusht_runner(
        cfg,
        Path("/tmp/soda_eval_test"),
        n_test=1,
        n_test_vis=0,
        max_steps=10,
        test_start_seed=100000,
    )
    assert runner._inner.n_action_steps == 1


def test_hierarchical_policy_predict_raises():
    from soda.inference.hierarchical_controller import HierarchicalPolicyConfig

    class _Device:
        type = "cpu"

    policy = HierarchicalPolicy(
        device=_Device(),
        config=HierarchicalPolicyConfig(),
    )
    try:
        policy.predict_action({})
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_slice_action_chunk_matches_dp():
    import numpy as np

    action = np.zeros((2, 8, 2), dtype=np.float32)
    out = slice_action_chunk(action, 1)
    assert out.shape == (2, 1, 2)
