"""Training losses for π_low (diffusion + termination BCE)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from soda.models.termination_head import termination_bce_from_logits


def diffusion_loss(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked MSE (DP ``compute_loss`` pattern)."""
    if mask is None:
        return F.mse_loss(model_pred, target)
    loss = F.mse_loss(model_pred, target, reduction="none")
    loss = loss * mask.type(loss.dtype)
    return loss[mask].mean()


def diffusion_loss_per_sample(
    model_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-batch-item masked MSE, shape ``(B,)``."""
    loss = F.mse_loss(model_pred, target, reduction="none")
    if mask is not None:
        loss = loss * mask.type(loss.dtype)
    batch_size = int(model_pred.shape[0])
    return loss.reshape(batch_size, -1).mean(dim=-1)


def termination_bce_loss(
    beta_logit: torch.Tensor,
    beta_label: torch.Tensor,
    *,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE on termination logits vs ``beta_label`` ``{0, 1}``."""
    return termination_bce_from_logits(
        beta_logit, beta_label, pos_weight=pos_weight
    )


def termination_bce_loss_per_sample(
    beta_logit: torch.Tensor,
    beta_label: torch.Tensor,
    *,
    pos_weight: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-batch-item BCE, shape ``(B,)``."""
    if beta_logit.ndim > 1:
        beta_logit = beta_logit.squeeze(-1)
    kwargs: dict[str, torch.Tensor] = {}
    if pos_weight is not None:
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(
                float(pos_weight), device=beta_logit.device, dtype=beta_logit.dtype
            )
        kwargs["pos_weight"] = pos_weight.reshape(1)
    return F.binary_cross_entropy_with_logits(
        beta_logit, beta_label.float(), reduction="none", **kwargs
    )


@dataclass(frozen=True)
class BetaConfusionCounts:
    """Batch-aggregatable confusion counts for β classification metrics."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    n_pred_ge: int = 0
    tp_pred_ge: int = 0

    def __add__(self, other: BetaConfusionCounts) -> BetaConfusionCounts:
        return BetaConfusionCounts(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            n_pred_ge=self.n_pred_ge + other.n_pred_ge,
            tp_pred_ge=self.tp_pred_ge + other.tp_pred_ge,
        )


def beta_confusion_counts_from_logits(
    beta_logit: torch.Tensor,
    beta_label: torch.Tensor,
    *,
    threshold: float = 0.5,
    high_prob_threshold: float = 0.9,
) -> BetaConfusionCounts:
    """
    Confusion counts for β=0/1 labels vs sigmoid(logit).

    ``high_prob_threshold`` supports precision@0.9: among preds with p >= 0.9,
    how many are true β=1 (``tp_pred_ge / n_pred_ge``).
    """
    prob = torch.sigmoid(beta_logit.reshape(-1))
    label = beta_label.reshape(-1).float()
    pred_pos = prob >= threshold
    label_pos = label >= 0.5

    tp = int((pred_pos & label_pos).sum().item())
    tn = int((~pred_pos & ~label_pos).sum().item())
    fp = int((pred_pos & ~label_pos).sum().item())
    fn = int((~pred_pos & label_pos).sum().item())

    high = prob >= high_prob_threshold
    n_pred_ge = int(high.sum().item())
    tp_pred_ge = int((high & label_pos).sum().item())
    return BetaConfusionCounts(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        n_pred_ge=n_pred_ge,
        tp_pred_ge=tp_pred_ge,
    )


def beta_metrics_from_counts(
    counts: BetaConfusionCounts,
    *,
    high_prob_threshold: float = 0.9,
) -> dict[str, float]:
    """Scalar β metrics for logging (W&B / metrics.json)."""
    n = counts.tp + counts.tn + counts.fp + counts.fn
    n_pos = counts.tp + counts.fn
    n_neg = counts.tn + counts.fp

    acc = (counts.tp + counts.tn) / n if n > 0 else float("nan")
    acc_pos = counts.tp / n_pos if n_pos > 0 else float("nan")
    acc_neg = counts.tn / n_neg if n_neg > 0 else float("nan")
    precision = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) > 0 else float("nan")
    recall = counts.tp / n_pos if n_pos > 0 else float("nan")
    precision_at = (
        counts.tp_pred_ge / counts.n_pred_ge if counts.n_pred_ge > 0 else float("nan")
    )

    return {
        "beta_acc": float(acc),
        "beta_acc_pos": float(acc_pos),
        "beta_acc_neg": float(acc_neg),
        "beta_precision": float(precision),
        "beta_recall": float(recall),
        f"beta_precision_at_{str(high_prob_threshold).replace('.', '_')}": float(
            precision_at
        ),
    }


def low_policy_total_loss(
    diffusion: torch.Tensor,
    termination: torch.Tensor,
    *,
    termination_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combine scalar diffusion + termination losses for ``backward()``.

    Returns
    -------
    loss_total
        ``L_diffusion + termination_weight * L_termination`` — the only tensor to backprop.
    log_dict
        Detached scalars for W&B / stdout: total plus each component separately.
    """
    total = diffusion + float(termination_weight) * termination
    return total, {
        "loss": float(total.detach().item()),
        "loss_diffusion": float(diffusion.detach().item()),
        "loss_termination": float(termination.detach().item()),
    }
