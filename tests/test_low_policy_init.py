"""Tests for π_low scaffolding (no full DP stack required for shape-meta helpers)."""

from __future__ import annotations

import pytest

from soda.models.low_policy import _patch_shape_meta_duration_channel
from soda.models.termination_head import TerminationHead, TerminationHeadConfig


def test_require_obs_as_global_cond_false_raises():
    from soda.models.low_policy import require_obs_as_global_cond

    assert require_obs_as_global_cond(True) is True
    with pytest.raises(ValueError, match="obs_as_global_cond"):
        require_obs_as_global_cond(False)


def test_patch_shape_meta_adds_duration_channel():
    meta = {
        "action": {"shape": [2]},
        "obs": {
            "image": {"shape": [3, 96, 96], "type": "rgb"},
            "agent_pos": {"shape": [2], "type": "low_dim"},
        },
    }
    out = _patch_shape_meta_duration_channel(meta)
    assert out["action"]["shape"] == (3,)
    assert meta["action"]["shape"] == [2]


def test_termination_head_forward_shape():
    head = TerminationHead(
        TerminationHeadConfig(bottleneck_dim=8, hidden_dim=16, num_layers=2)
    )
    import torch

    bn = torch.randn(4, 8)
    logit = head(bn)
    assert logit.shape == (4,)


def _dp_stack_available() -> bool:
    from pathlib import Path

    if not Path("third_party/diffusion_policy/eval.py").is_file():
        return False
    try:
        import einops  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _dp_stack_available(),
    reason="diffusion_policy submodule or DP deps (einops, robomimic, …) not available",
)
def test_low_policy_is_dp_subclass():
    from soda.models.low_policy import LowPolicy
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    assert issubclass(LowPolicy, DiffusionUnetHybridImagePolicy)
