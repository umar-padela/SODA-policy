"""
Termination head β_ω(s) for π_low (project_plan §2, §7 row 19).

``beta = MLP(stop_grad(bottleneck))`` — BCE at the dataset anchor frame; no gradient
from β into the diffusion U-Net (locked in §8).

Used by ``LowPolicy`` and ``losses_low.termination_bce_loss``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TerminationHeadConfig:
    """
    MLP hyperparameters (populate from Hydra ``termination_head:`` or nested in ``low_policy:``).

    All fields required — no silent defaults (set in ``configs/pusht/soda_*.yaml``).
    """

    bottleneck_dim: int
    hidden_dim: int
    num_layers: int


class TerminationHead(nn.Module):
    """
    Predict per-step termination logit from a **stopped** U-Net bottleneck.

    Training
    --------
    ``LowPolicy`` passes ``bottleneck.detach()`` (or this module applies ``stop_grad``)
    so ``L_termination`` does not update diffusion weights.

    Inference
    ---------
    ``HierarchicalController`` compares ``sigmoid(logit)`` to ``beta_transition`` (§8E).
    """

    def __init__(self, cfg: TerminationHeadConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {cfg.num_layers}")

        layers: list[nn.Module] = [
            nn.Linear(cfg.bottleneck_dim, cfg.hidden_dim),
            nn.Mish(),
        ]
        for _ in range(cfg.num_layers - 1):
            layers.extend(
                [
                    nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                    nn.Mish(),
                ]
            )
        layers.append(nn.Linear(cfg.hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, bottleneck: torch.Tensor, *, stop_grad: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        bottleneck
            U-Net bottleneck features ``(B, bottleneck_dim)``.
        stop_grad
            If True (default), apply ``bottleneck.detach()`` before MLP (locked §8).

        Returns
        -------
        logit
            ``(B,)`` pre-sigmoid termination logit.
        """
        feat = bottleneck.detach() if stop_grad else bottleneck
        return self.mlp(feat).squeeze(-1)

    def predict_beta(self, bottleneck: torch.Tensor) -> torch.Tensor:
        """Termination probability in ``(0, 1)`` for inference / logging."""
        return torch.sigmoid(self.forward(bottleneck, stop_grad=True))


def termination_bce_from_logits(
    logit: torch.Tensor,
    beta_label: torch.Tensor,
    *,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE with logits (shared helper for ``TerminationHead`` and ``losses_low``)."""
    if logit.ndim > 1:
        logit = logit.squeeze(-1)
    kwargs: dict[str, torch.Tensor] = {}
    if pos_weight is not None:
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(float(pos_weight), device=logit.device, dtype=logit.dtype)
        kwargs["pos_weight"] = pos_weight.reshape(1)
    return F.binary_cross_entropy_with_logits(logit, beta_label.float(), **kwargs)
