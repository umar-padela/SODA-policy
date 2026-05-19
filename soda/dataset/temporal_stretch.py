"""
Variable-horizon resampling for SODA low-level training (project_plan §2, §7 row 10).

Maps option segments of length ``h_i`` to a fixed training length ``h_max`` via linear
interpolation, and reverses the mapping at inference using a predicted horizon channel.

References
----------
- ``project_proposal.md`` — Variable-Horizon Action Chunking via Temporal Stretching
- ``project_plan.md`` §2 (horizon decode locked: mean-pool → ``unstretch``)

Implementation TODO (§7 row 10)
-------------------------------
- [ ] ``TemporalStretcher.__init__``: store ``h_max``, interpolation mode (linear default),
      optional ``eps`` for degenerate length-1 segments.
- [ ] ``stretch(actions, h_orig)`` — resample one segment ``actions`` shape ``(h_i, D)`` to
      ``(h_max, D)``; return stretched actions + normalized horizon target for the
      ``D+1`` channel (e.g. ``h_orig / h_max`` broadcast on last dim).
- [ ] ``stretch_batch`` — vectorized or looped helper for collate / dataset use.
- [ ] ``unstretch(actions, h_pred)`` — inverse resample ``(h_max, D)`` → ``(h_pred, D)``
      with ``h_pred`` from mean-pooled horizon channel (integer clip to ``[1, h_max]``).
- [ ] ``decode_horizon(action_chunk)`` — ``h_pred = mean(action_chunk[..., -1])`` then
      denormalize to step count (locked in §2); used by ``low_policy`` at inference.
- [ ] Unit tests: round-trip sanity (stretch → unstretch with true ``h_i``); edge cases
      ``h_i=1``, ``h_i=h_max``; add ``tests/test_temporal_stretch.py``.
- [ ] Document whether stretching applies to **actions only** at train time or also
      obs windows (likely actions + parallel obs indexing in dataset, not inside this class).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalStretcher:
    """
    Linear temporal resampling between native segment length and fixed ``h_max``.

    Parameters
    ----------
    h_max
        Fixed sequence length for the 1D U-Net (DP ``horizon`` / action chunk size).
    """

    h_max: int

    def stretch(
        self,
        actions: np.ndarray,
        *,
        h_orig: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Resample a variable-length action segment to length ``h_max``.

        Parameters
        ----------
        actions
            Shape ``(h_i, D)`` native expert actions for one option segment.
        h_orig
            Original length ``h_i``; if ``None``, inferred from ``actions.shape[0]``.

        Returns
        -------
        stretched
            Shape ``(h_max, D)``.
        horizon_normalized
            Scalar in ``(0, 1]`` for the horizon channel target (e.g. ``h_i / h_max``).
        """
        raise NotImplementedError("TODO §7 row 10: linear interpolate actions to h_max")

    def unstretch(
        self,
        actions: np.ndarray,
        h_pred: int | float,
    ) -> np.ndarray:
        """
        Resample a fixed-length chunk back to ``h_pred`` steps for execution.

        Parameters
        ----------
        actions
            Shape ``(h_max, D)`` (may include horizon channel in last dim at call site).
        h_pred
            Predicted segment length (steps), typically from ``decode_horizon``.
        """
        raise NotImplementedError("TODO §7 row 10: inverse resample to h_pred steps")

    @staticmethod
    def decode_horizon(action_chunk: np.ndarray) -> int:
        """
        Decode predicted horizon from ``D+1`` action chunk (mean-pool last channel).

        Parameters
        ----------
        action_chunk
            Shape ``(..., D+1)`` with horizon in ``[..., -1]``.

        Returns
        -------
        int
            Clipped step count in ``[1, h_max]`` (caller passes ``h_max`` for clipping).
        """
        raise NotImplementedError("TODO §7 row 10: mean-pool horizon channel → int steps")
