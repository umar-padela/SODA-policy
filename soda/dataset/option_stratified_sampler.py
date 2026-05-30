"""
Stratified mini-batches for π_low termination (β) training.

Two samplers are provided:

- :class:`OptionBetaStratifiedBatchSampler` — legacy; works with the random-anchor
  (1 entry per segment) dataset mode and encoded segment indices.
- :class:`AllAnchorsBetaStratifiedBatchSampler` — for ``all_anchors=True`` datasets;
  takes explicit pos/neg flat index lists and guarantees the natural-rate β=1 fraction
  per batch (e.g. 4/64 for Push-T at 5.6% pos rate).

Default positive fraction: ``1 / (gamma + 1)`` where ``gamma`` is BCE ``pos_weight``.
That balances loss mass: ``f * gamma ≈ (1 - f)`` for per-sample BCE with pos_weight.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


def encode_stratified_index(segment_index: int, *, at_segment_end: bool) -> int:
    """Map ``(segment_index, anchor kind)`` → dataset index."""
    return int(segment_index) * 2 + (1 if at_segment_end else 0)


def decode_stratified_index(index: int) -> tuple[int, bool]:
    """Inverse of :func:`encode_stratified_index`."""
    return int(index) // 2, (int(index) % 2) == 1


def beta_batch_pos_fraction_from_gamma(gamma: float) -> float:
    """
    Target β=1 fraction per mini-batch given BCE ``pos_weight`` γ.

    Solves ``f * γ = (1 - f)`` → ``f = 1 / (γ + 1)``.  (Not ``1/γ``.)
    """
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    return 1.0 / (gamma + 1.0)


def positives_per_batch(batch_size: int, pos_fraction: float) -> int:
    """At least one β=1 and one β=0 sample per batch when ``batch_size >= 2``."""
    if batch_size < 2:
        raise ValueError(f"batch_size must be >= 2 for stratified β batches, got {batch_size}")
    frac = float(pos_fraction)
    if not 0.0 < frac < 1.0:
        raise ValueError(f"pos_fraction must be in (0, 1), got {frac}")
    n_pos = int(round(batch_size * frac))
    n_pos = max(1, min(batch_size - 1, n_pos))
    return n_pos


class AllAnchorsBetaStratifiedBatchSampler(Sampler[list[int]]):
    """
    Stratified batch sampler for ``all_anchors=True`` datasets.

    Takes explicit ``pos_indices`` / ``neg_indices`` lists (flat dataset indices
    where beta_label == 1 / 0) and guarantees ``n_pos = max(1, round(batch_size ×
    pos_rate))`` β=1 examples per batch, where ``pos_rate = len(pos) / total``.

    For Push-T with 5.6% pos rate and ``batch_size=64``, this gives 4 β=1 per batch
    regardless of how the epoch shuffle falls.  Positive pool wraps with replacement
    when ``n_batches * n_pos_per_batch > len(pos_indices)``; negative pool likewise.
    """

    def __init__(
        self,
        pos_indices: list[int],
        neg_indices: list[int],
        batch_size: int,
        *,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if not pos_indices:
            raise ValueError("pos_indices must be non-empty")
        if not neg_indices:
            raise ValueError("neg_indices must be non-empty")
        self.pos_indices = list(pos_indices)
        self.neg_indices = list(neg_indices)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        total = len(self.pos_indices) + len(self.neg_indices)
        pos_rate = len(self.pos_indices) / total
        self.pos_per_batch = positives_per_batch(batch_size, pos_rate)
        self.neg_per_batch = self.batch_size - self.pos_per_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = len(self.pos_indices) + len(self.neg_indices)
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n_batches = len(self)
        n_pos_needed = n_batches * self.pos_per_batch
        n_neg_needed = n_batches * self.neg_per_batch

        pos_arr = rng.permutation(np.asarray(self.pos_indices))
        neg_arr = rng.permutation(np.asarray(self.neg_indices))

        if n_pos_needed > len(pos_arr):
            reps = math.ceil(n_pos_needed / len(pos_arr))
            pos_arr = np.concatenate([pos_arr] * reps)
        pos_pool = pos_arr[:n_pos_needed]

        if n_neg_needed > len(neg_arr):
            reps = math.ceil(n_neg_needed / len(neg_arr))
            neg_arr = np.concatenate([neg_arr] * reps)
        neg_pool = neg_arr[:n_neg_needed]

        for batch_idx in range(n_batches):
            ps = batch_idx * self.pos_per_batch
            ns = batch_idx * self.neg_per_batch
            pos_pick = pos_pool[ps : ps + self.pos_per_batch]
            neg_pick = neg_pool[ns : ns + self.neg_per_batch]
            batch = [int(x) for x in pos_pick] + [int(x) for x in neg_pick]
            rng.shuffle(batch)
            yield batch


class OptionBetaStratifiedBatchSampler(Sampler[list[int]]):
    """
    Yields batches of encoded dataset indices (see :func:`encode_stratified_index`).

    Same number of batches per epoch as ``num_segments // batch_size`` with
    ``drop_last=True`` (matches the default π_low train loader).
    """

    def __init__(
        self,
        num_segments: int,
        batch_size: int,
        pos_fraction: float,
        *,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_segments < 1:
            raise ValueError("num_segments must be positive")
        self.num_segments = int(num_segments)
        self.batch_size = int(batch_size)
        self.pos_fraction = float(pos_fraction)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.pos_per_batch = positives_per_batch(batch_size, pos_fraction)
        self.neg_per_batch = self.batch_size - self.pos_per_batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_segments // self.batch_size
        return (self.num_segments + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        pos_pool = rng.permutation(
            [encode_stratified_index(i, at_segment_end=True) for i in range(self.num_segments)]
        )
        neg_pool = rng.permutation(
            [encode_stratified_index(i, at_segment_end=False) for i in range(self.num_segments)]
        )

        n_batches = len(self)
        for batch_idx in range(n_batches):
            pos_start = batch_idx * self.pos_per_batch
            neg_start = batch_idx * self.neg_per_batch
            pos_pick = pos_pool[
                pos_start : pos_start + self.pos_per_batch
            ]
            if len(pos_pick) < self.pos_per_batch:
                extra = rng.choice(
                    pos_pool,
                    size=self.pos_per_batch - len(pos_pick),
                    replace=True,
                )
                pos_pick = np.concatenate([pos_pick, extra])

            neg_pick = neg_pool[
                neg_start : neg_start + self.neg_per_batch
            ]
            if len(neg_pick) < self.neg_per_batch:
                extra = rng.choice(
                    neg_pool,
                    size=self.neg_per_batch - len(neg_pick),
                    replace=True,
                )
                neg_pick = np.concatenate([neg_pick, extra])

            batch = [int(x) for x in pos_pick] + [int(x) for x in neg_pick]
            rng.shuffle(batch)
            yield batch
