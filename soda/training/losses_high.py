"""
Training losses for π_high (high-level flow matching on options) — project_plan §7 row 15.

Used by ``train_high.py`` with ``HighPolicy``, ``OptionStartDataset``.

Pattern copied from ``assignment1/hw1/losses.py::flow_matching_loss``:
    t ~ U[0,1]  →  interpolate(x1, t)  →  predict_velocity  →  MSE(v_pred, v_tgt)

Symmetric with ``losses_low.py`` (π_low diffusion + termination).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from soda.models.high_policy import OptionFlowMatchingSchedule

if TYPE_CHECKING:
    from soda.models.high_policy import HighPolicy


def flow_matching_option_loss(
    policy: "HighPolicy",
    global_feat: torch.Tensor,
    option_id: torch.Tensor,
) -> torch.Tensor:
    """
    Flow-matching loss on option embeddings (HW1 ``flow_matching_loss`` analogue).

    Parameters
    ----------
    policy
        ``HighPolicy`` with ``option_target``, ``predict_velocity``, and ``schedule``.
    global_feat
        ``(B, global_feat_dim)`` from ``encode_obs`` on segment-start batches.
    option_id
        ``(B,)`` long — ground-truth ω per segment.

    Returns
    -------
    Scalar MSE loss (mean over batch).
    """
    x1 = policy.option_target(option_id)
    B = option_id.shape[0]
    t = torch.rand(B, device=option_id.device, dtype=x1.dtype)
    x_t, v_tgt = OptionFlowMatchingSchedule.interpolate(x1, t)
    v_pred = policy.predict_velocity(x_t, global_feat, t)
    return F.mse_loss(v_pred, v_tgt)
