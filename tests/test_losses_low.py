"""Unit tests for π_low training losses."""

from __future__ import annotations

import torch

from soda.training.losses_low import (
    diffusion_loss,
    low_policy_total_loss,
    termination_bce_loss,
)


def test_diffusion_loss_masked():
    pred = torch.randn(2, 4, 3)
    target = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    mask[:, 0, :] = False
    loss = diffusion_loss(pred, target, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_termination_bce_loss():
    logit = torch.tensor([2.0, -2.0])
    label = torch.tensor([1.0, 0.0])
    loss = termination_bce_loss(logit, label)
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_termination_bce_loss_pos_weight_increases_positive_penalty():
    logit = torch.tensor([-2.0, -2.0])
    label = torch.tensor([1.0, 0.0])
    base = termination_bce_loss(logit, label)
    weighted = termination_bce_loss(logit, label, pos_weight=10.0)
    assert weighted.item() > base.item()


def test_low_policy_total_loss():
    total, logs = low_policy_total_loss(
        torch.tensor(1.0), torch.tensor(0.5), termination_weight=2.0
    )
    assert total.item() == 2.0
    assert logs["loss"] == 2.0
    assert logs["loss_diffusion"] == 1.0
    assert logs["loss_termination"] == 0.5


def test_beta_metrics_from_counts_perfect_and_precision_at():
    from soda.training.losses_low import (
        BetaConfusionCounts,
        beta_confusion_counts_from_logits,
        beta_metrics_from_counts,
    )

    logit = torch.tensor([5.0, -5.0, 5.0, -5.0])
    label = torch.tensor([1.0, 0.0, 1.0, 0.0])
    counts = beta_confusion_counts_from_logits(logit, label, threshold=0.5)
    metrics = beta_metrics_from_counts(counts, high_prob_threshold=0.9)
    assert metrics["beta_acc"] == 1.0
    assert metrics["beta_acc_pos"] == 1.0
    assert metrics["beta_acc_neg"] == 1.0
    assert metrics["beta_precision_at_0_9"] == 1.0

    noisy = beta_confusion_counts_from_logits(
        torch.tensor([5.0, 5.0, -5.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        threshold=0.5,
        high_prob_threshold=0.9,
    )
    noisy_metrics = beta_metrics_from_counts(noisy, high_prob_threshold=0.9)
    assert noisy_metrics["beta_acc_neg"] == 0.5   # 1 TN out of 2 negatives (1 FP)
    assert noisy_metrics["beta_acc_pos"] == 1.0   # 1 TP out of 1 positive
    assert noisy_metrics["beta_precision_at_0_9"] == 0.5
