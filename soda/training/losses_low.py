"""
Training losses for π_low (low-level diffusion + termination) — project_plan §7 row 14.

Used by ``train_low.py`` with ``LowPolicy`` and ``OptionLabeledZarrDataset``.

Symmetric with ``losses_high.py`` (π_high flow matching).
"""

from __future__ import annotations

from typing import Any

# TODO §7 row 14: import torch, torch.nn.functional as F


def diffusion_loss(
    model_pred: Any,
    target: Any,
    mask: Any | None = None,
) -> Any:
    """
    Diffusion Policy-style noise / sample MSE on action chunks.

    TODO
    ----
    - [ ] Match DP ``compute_loss`` interface from the workspace you subclass
    - [ ] Support ``(B, T, Da)`` with optional inpainting mask
    - [ ] Action dim includes duration channel (D+1) when training π_low
    """
    raise NotImplementedError("TODO §7 row 14: diffusion_loss")


def termination_bce_loss(
    beta_pred: Any,
    beta_label: Any,
) -> Any:
    """
    BCE between termination head output and per-frame ``beta_label``.

    TODO
    ----
    - [ ] ``beta_pred`` shape ``(B,)`` or ``(B, 1)``; labels ``(B,)`` float 0/1
    - [ ] Confirm ``beta_label`` comes from dataset anchor frame (see dataset doc)
    """
    raise NotImplementedError("TODO §7 row 14: termination_bce_loss")


def low_policy_total_loss(
    diffusion: Any,
    termination: Any,
    *,
    termination_weight: float = 1.0,
) -> tuple[Any, dict[str, float]]:
    """
    ``L_total = L_diffusion + termination_weight * L_termination``.

    TODO
    ----
    - [ ] Return scalar loss + dict of detached floats for logging
    - [ ] Project plan: no λ sweep required; optional weight=1.0 default
    """
    raise NotImplementedError("TODO §7 row 14: low_policy_total_loss")
