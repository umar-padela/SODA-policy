"""Capture U-Net mid-layer (bottleneck) without editing ``third_party/diffusion_policy``."""

from __future__ import annotations

from typing import Any

import torch


def pool_bottleneck(bottleneck: torch.Tensor) -> torch.Tensor:
    """``(B, C, T)`` → ``(B, C)`` mean over horizon."""
    if bottleneck.ndim != 3:
        raise ValueError(f"Expected (B, C, T), got {tuple(bottleneck.shape)}")
    return bottleneck.mean(dim=-1)


def forward_unet_with_bottleneck(
    model: torch.nn.Module,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    global_cond: torch.Tensor,
    *,
    local_cond: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One U-Net forward; return ``(prediction, pooled_bottleneck)``."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
        captured["bn"] = output

    handle = model.mid_modules[-1].register_forward_hook(hook)
    try:
        pred = model(
            sample,
            timestep,
            local_cond=local_cond,
            global_cond=global_cond,
        )
    finally:
        handle.remove()

    if "bn" not in captured:
        raise RuntimeError("U-Net mid hook did not capture bottleneck")
    return pred, pool_bottleneck(captured["bn"])
