"""Inverse-frequency class weights for option-segment training losses."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def option_ids_from_segments(segments: Sequence[object]) -> np.ndarray:
    """Extract ``option_id`` from ``OptionSegment``-like objects."""
    return np.asarray([int(seg.option_id) for seg in segments], dtype=np.int64)


def count_option_ids(option_ids: np.ndarray, num_options: int) -> np.ndarray:
    """Per-class segment counts, length ``num_options``."""
    ids = np.asarray(option_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        raise ValueError("option_ids must be non-empty")
    return np.bincount(ids, minlength=int(num_options)).astype(np.int64)


def compute_inverse_freq_class_weights(
    option_ids: np.ndarray,
    num_options: int,
) -> torch.Tensor:
    """
    Per-class loss multipliers: ``weight[c] = N / (K * count[c])``.

    Each class contributes equally in expectation while every unique segment is
    still visited once per epoch (loss weighting, not oversampling).
    Weights are normalized so their count-weighted mean is 1.0.
    """
    counts = count_option_ids(option_ids, num_options).astype(np.float64)
    if np.any(counts <= 0):
        missing = np.flatnonzero(counts <= 0).tolist()
        raise ValueError(
            f"Cannot balance options: classes {missing} have zero train segments"
        )
    total = float(counts.sum())
    k = float(num_options)
    weights = total / (k * counts)
    weights = weights * (total / float(weights @ counts))
    return torch.tensor(weights, dtype=torch.float32)


def weighted_option_mean(
    per_sample_loss: torch.Tensor,
    option_id: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean of ``per_sample_loss`` with inverse-frequency class weights."""
    ids = option_id.reshape(-1).long()
    w = class_weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)[ids]
    return (per_sample_loss.reshape(-1) * w).mean()


def format_option_weight_summary(
    counts: np.ndarray,
    class_weights: torch.Tensor,
    *,
    skill_names: dict[int, str] | None = None,
) -> str:
    """Human-readable table for train startup logs."""
    lines = ["Option inverse-frequency loss weights (train only):"]
    lines.append(f"{'Option':<15} | {'Segments':<10} | {'Weight':<8}")
    lines.append("-" * 38)
    for c in range(int(counts.shape[0])):
        label = skill_names.get(c, str(c)) if skill_names else str(c)
        lines.append(
            f"{label:<15} | {int(counts[c]):<10} | {class_weights[c].item():>8.4f}"
        )
    return "\n".join(lines)


def compute_termination_pos_weight(
    option_ids: np.ndarray,
    episode_ends: np.ndarray,
    train_episode_mask: np.ndarray,
) -> tuple[float, int, int]:
    """
    ``gamma = (# beta=0 frames) / (# beta=1 frames)`` on train episodes.

    One positive beta label per option segment end (``derive_beta_labels``).
    """
    from soda.dataset.option_aware_dataset import derive_beta_labels

    beta = derive_beta_labels(option_ids, episode_ends)
    frame_mask = np.zeros(beta.shape[0], dtype=bool)
    start = 0
    for ep_idx, ep_end in enumerate(episode_ends):
        if bool(train_episode_mask[ep_idx]):
            frame_mask[start:ep_end] = True
        start = int(ep_end)

    beta_train = beta[frame_mask]
    n_pos = int(beta_train.sum())
    if n_pos == 0:
        raise ValueError("No positive beta labels in train episodes")
    n_neg = int(len(beta_train) - n_pos)
    return float(n_neg) / float(n_pos), n_neg, n_pos


def format_termination_pos_weight_summary(
    *,
    gamma: float,
    n_neg: int,
    n_pos: int,
    manual: bool = False,
) -> str:
    source = "manual" if manual else "auto (neg frames / pos frames on train episodes)"
    lines = [
        "Termination BCE pos_weight (train only; PyTorch BCE pos_weight on beta=1):",
        f"  {source}, gamma={gamma:.4f}, negatives={n_neg}, positives={n_pos}",
    ]
    return "\n".join(lines)


def resolve_termination_pos_weight(
    *,
    configured: float | None,
    option_ids: np.ndarray,
    episode_ends: np.ndarray,
    train_episode_mask: np.ndarray,
) -> tuple[float, int, int, bool]:
    """Resolve ``termination_pos_weight`` from yaml or train beta label counts."""
    if configured is not None:
        return float(configured), 0, 0, True
    gamma, n_neg, n_pos = compute_termination_pos_weight(
        option_ids, episode_ends, train_episode_mask
    )
    return gamma, n_neg, n_pos, False


def resolve_option_class_weights(
    mode: str | None,
    option_ids: np.ndarray,
    num_options: int,
) -> torch.Tensor | None:
    """Parse ``option_balance`` config: ``none`` or ``inverse_freq``."""
    if mode is None or str(mode).lower() in {"none", "false", "0", ""}:
        return None
    if str(mode).lower() in {"inverse_freq", "inverse_frequency"}:
        return compute_inverse_freq_class_weights(option_ids, num_options)
    raise ValueError(
        f"Unknown option_balance={mode!r}; expected 'none' or 'inverse_freq'"
    )


def resolve_beta_batch_pos_fraction(
    configured: float | None,
    *,
    termination_pos_weight: float,
) -> float:
    """
    Target β=1 fraction per stratified mini-batch.

    Default ``1 / (gamma + 1)`` so ``f * gamma ≈ (1 - f)`` in weighted BCE.
    """
    if configured is not None:
        frac = float(configured)
    else:
        from soda.dataset.option_stratified_sampler import beta_batch_pos_fraction_from_gamma

        frac = beta_batch_pos_fraction_from_gamma(termination_pos_weight)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"beta_batch_pos_fraction must be in (0, 1), got {frac}")
    return frac
