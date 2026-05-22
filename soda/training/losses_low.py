"""
Training losses for π_low (low-level diffusion + termination) — project_plan §7 rows 18–19.

Used by ``train_low.py`` with ``LowPolicy`` and ``OptionLabeledZarrDataset``.

Upstream reference: ``diffusion_policy`` workspace ``compute_loss`` on
``DiffusionUnetHybridImagePolicy`` (noise / sample MSE with optional mask).

Symmetric with ``losses_high.py`` (π_high flow matching).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from soda.models.low_policy import LowPolicy


def diffusion_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Diffusion Policy-style noise / sample MSE on action chunks.

    Typically called **inside** ``LowPolicy.compute_loss`` after the parent DP step
    produces ``model_pred`` and diffusion target (epsilon or x0 per scheduler).

    Parameters
    ----------
    model_pred
        U-Net output, shape matches target (often ``(B, T, Da+1)``).
    target
        Training target from noise scheduler (same shape).
    mask
        Optional inpainting / loss mask (DP ``compute_loss`` pattern).

    Returns
    -------
    Scalar mean MSE over masked elements.

    TODO §7 row 18
    --------------
    - [ ] If ``mask is None``: ``return F.mse_loss(model_pred, target)``
    - [ ] Else: elementwise MSE, multiply by mask, mean over active entries
    - [ ] Match dtype/device of ``model_pred``
    - [ ] **Do not** apply termination weight here — keep diffusion term separate
    """
    raise NotImplementedError("TODO §7 row 18: diffusion_loss")


def termination_bce_loss(
    beta_pred: torch.Tensor,
    beta_label: torch.Tensor,
) -> torch.Tensor:
    """
    BCE between termination head output and per-frame ``beta_label``.

    Prefer ``termination_head.termination_bce_from_logits`` if ``beta_pred`` is a logit.

    Parameters
    ----------
    beta_pred
        ``(B,)`` or ``(B, 1)`` logits **or** probabilities (document which in impl).
    beta_label
        ``(B,)`` float in ``{0, 1}`` from dataset anchor frame (segment-end frame).

    Returns
    -------
    Scalar BCE loss (mean over batch).

    TODO §7 row 19
    --------------
    - [ ] If logits: ``F.binary_cross_entropy_with_logits``
    - [ ] Labels ``beta_label.float()`` on same device
    - [ ] Confirm dataset supplies ``beta_label`` only on frames where termination is defined
    """
    raise NotImplementedError("TODO §7 row 19: termination_bce_loss")


def low_policy_total_loss(
    diffusion: torch.Tensor,
    termination: torch.Tensor,
    *,
    termination_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    ``L_total = L_diffusion + termination_weight * L_termination`` (project_plan §2).

    Parameters
    ----------
    diffusion
        Scalar ``L_diffusion`` (grad flows to U-Net + option embed + obs encoder per DP).
    termination
        Scalar ``L_termination`` (grad flows **only** to ``TerminationHead`` if bottleneck detached).
    termination_weight
        Default ``1.0`` — project plan: no λ sweep required.

    Returns
    -------
    loss
        Scalar for ``loss.backward()``.
    log_dict
        Detached floats, e.g. ``{"loss": ..., "loss_diffusion": ..., "loss_termination": ...}``.

    TODO §7 row 18
    --------------
    - [ ] ``total = diffusion + termination_weight * termination``
    - [ ] Build log dict from ``.detach().item()`` for each component
    """
    raise NotImplementedError("TODO §7 row 18: low_policy_total_loss")


def compute_low_policy_loss(
    policy: "LowPolicy",
    batch: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Thin wrapper: full π_low forward + loss aggregation.

    Alternative to implementing everything inside ``LowPolicy.compute_loss``;
    pick **one** call site (policy method vs this helper) to avoid duplication.

    TODO §7 row 18
    --------------
    - [ ] ``return policy.compute_loss(batch)`` once ``LowPolicy`` is implemented
    """
    raise NotImplementedError("TODO §7 row 18: compute_low_policy_loss")
