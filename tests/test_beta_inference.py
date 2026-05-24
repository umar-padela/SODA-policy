"""Tests for β inference helpers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from soda.models.unet_bottleneck import forward_unet_with_bottleneck, pool_bottleneck


def test_pool_bottleneck_mean_over_time():
    bn = torch.randn(2, 8, 5)
    out = pool_bottleneck(bn)
    assert out.shape == (2, 8)
    assert torch.allclose(out, bn.mean(dim=-1))


def test_decode_segment_steps():
    from soda.dataset.temporal_stretch import decode_segment_steps as decode_np
    from soda.models.low_policy import decode_segment_steps as decode_torch

    chunk = torch.zeros(16, 3)
    chunk[:, -1] = 0.5
    assert decode_np(chunk.numpy(), 16) == 8
    assert decode_torch(chunk, 16) == 8.0

    batch = torch.zeros(2, 16, 3)
    batch[0, :, -1] = 0.25
    batch[1, :, -1] = 1.0
    assert decode_torch(batch, 16).tolist() == [4, 16]


def _dp_stack_available() -> bool:
    from pathlib import Path

    if not Path("third_party/diffusion_policy/eval.py").is_file():
        return False
    try:
        import einops  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _dp_stack_available(), reason="DP deps not available")
def test_low_policy_decode_delegates():
    from soda.models.low_policy import LowPolicy, decode_segment_steps

    chunk = torch.zeros(16, 3)
    chunk[:, -1] = 0.5
    assert LowPolicy.decode_segment_steps(chunk, 16) == decode_segment_steps(chunk, 16)


class _TinyMidUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mid = nn.Conv1d(4, 8, kernel_size=1)
        self.mid_modules = nn.ModuleList([self.mid])
        self.out = nn.Conv1d(8, 4, kernel_size=1)

    def forward(self, sample, timestep, local_cond=None, global_cond=None):
        _ = timestep, local_cond, global_cond
        x = sample.transpose(1, 2)
        x = self.mid(x)
        x = self.out(x)
        return x.transpose(1, 2)


def test_forward_unet_with_bottleneck_hook():
    model = _TinyMidUnet()
    sample = torch.randn(2, 6, 4)
    t = torch.zeros(2, dtype=torch.long)
    cond = torch.randn(2, 3)
    pred, bn = forward_unet_with_bottleneck(model, sample, t, cond)
    assert pred.shape == sample.shape
    assert bn.shape == (2, 8)
