"""
Utilities for termination head training configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soda.dataset.option_aware_dataset import OptionAwareDataset


def compute_auto_pos_weight(dataset: "OptionAwareDataset", multiplier: float = 0.6) -> float:
    """
    Compute BCE pos_weight γ = avg_segment_len × multiplier.

    With Push-T supervised labels (avg segment ≈ 17 steps) and multiplier=0.6:
    γ ≈ 10, which compensates the ~17:1 class imbalance at ~60% of full balancing.
    Full balancing (multiplier=1.0) would over-fire β; 0.6 is a conservative default.

    Only applies to the β BCE loss (termination_pos_weight). Diffusion loss is
    not weighted (option_balance: none) so this does not affect diffusion training.
    """
    if not dataset.segments:
        raise ValueError("dataset has no segments")
    avg_len = sum(s.length for s in dataset.segments) / len(dataset.segments)
    return round(avg_len * multiplier, 1)
