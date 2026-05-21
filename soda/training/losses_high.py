"""
Training losses for π_high (high-level flow matching on options) — project_plan §7 row 15.

Used by ``train_high.py`` with ``HighPolicy`` and ``OptionStartDataset``.

Symmetric with ``losses_low.py`` (π_low diffusion + termination).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# TODO §7 row 15: import torch, torch.nn as nn, torch.nn.functional as F

if TYPE_CHECKING:
    from soda.models.high_policy import HighPolicy


def flow_matching_option_loss(
    policy: "HighPolicy",
    global_feat: Any,
    option_id: Any,
) -> Any:
    """
    Flow-matching loss: learn velocity field from noise → option embedding.

    Parameters
    ----------
    policy
        ``HighPolicy`` with flow network (may delegate to ``policy.compute_fm_loss``).
    global_feat
        ``(B, D)`` encoded segment-start observations.
    option_id
        ``(B,)`` long tensor of ground-truth ω per segment.

    TODO §7 row 15
    --------------
    - [ ] Option target: ``nn.Embedding`` lookup → ``x_1`` (continuous target space)
    - [ ] Sample ``t ~ U[0,1]``, noise ``x_0``, interpolate ``x_t`` (linear path)
    - [ ] Predict velocity ``v_pred``; target ``v_tgt = x_1 - x_0``
    - [ ] Return mean squared error over batch
    - [ ] Thin wrapper around ``policy.compute_fm_loss`` once implemented
    """
    raise NotImplementedError("TODO §7 row 15: flow_matching_option_loss")
