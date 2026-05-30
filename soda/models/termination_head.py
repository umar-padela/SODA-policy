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
    use_chunk_cursor: bool = False  # if True, concatenate a scalar cursor ∈ [0,1] to features
    dropout_rate: float = 0.0       # dropout after each hidden Mish; 0.0 = disabled


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

        input_dim = cfg.bottleneck_dim + (1 if cfg.use_chunk_cursor else 0)

        def _block(in_dim: int, out_dim: int) -> list[nn.Module]:
            mods: list[nn.Module] = [nn.Linear(in_dim, out_dim), nn.Mish()]
            if cfg.dropout_rate > 0.0:
                mods.append(nn.Dropout(cfg.dropout_rate))
            return mods

        layers: list[nn.Module] = _block(input_dim, cfg.hidden_dim)
        for _ in range(cfg.num_layers - 1):
            layers.extend(_block(cfg.hidden_dim, cfg.hidden_dim))
        layers.append(nn.Linear(cfg.hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        bottleneck: torch.Tensor,
        *,
        stop_grad: bool = True,
        chunk_cursor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        bottleneck
            Features ``(B, bottleneck_dim)`` — obs+option or U-Net bottleneck.
        stop_grad
            If True, detach features before MLP so β gradient does not reach the backbone.
        chunk_cursor
            Optional ``(B,)`` or ``(B, 1)`` scalar ∈ [0, 1]: normalized progress within
            the current option segment. Concatenated to features when ``use_chunk_cursor=True``.

        Returns
        -------
        logit
            ``(B,)`` pre-sigmoid termination logit.
        """
        feat = bottleneck.detach() if stop_grad else bottleneck
        if self.cfg.use_chunk_cursor and chunk_cursor is not None:
            cursor = chunk_cursor.to(dtype=feat.dtype, device=feat.device)
            if cursor.dim() == 1:
                cursor = cursor.unsqueeze(-1)
            feat = torch.cat([feat, cursor], dim=-1)
        return self.mlp(feat).squeeze(-1)

    def predict_beta(self, bottleneck: torch.Tensor, chunk_cursor: torch.Tensor | None = None) -> torch.Tensor:
        """Termination probability in ``(0, 1)`` for inference / logging."""
        return torch.sigmoid(self.forward(bottleneck, stop_grad=True, chunk_cursor=chunk_cursor))


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
