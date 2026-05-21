"""
Variable-horizon resampling for SODA low-level training (project_plan §2, §7 row 10).

Maps option segments of native length ``segment_steps`` to a fixed training length
``horizon_stretch_max`` via linear interpolation, and reverses the mapping at inference using
the predicted **duration channel** (mean-pool → ``unstretch``).

Naming (avoid confusion with eval control)
-----------------------------------------
- ``horizon_stretch_max``: training hyperparameter — resampled action chunk length for the
  low-level U-Net (often set to the longest skill in the dataset). **Not** used in sim.
- ``action_horizon``: eval-only — how many executed actions before replanning
  (``soda/inference/control_regimes.py``; P0 default ``1`` = closed-loop).
- ``policy_horizon``: frozen DP checkpoint prediction length (e.g. 16); unrelated to SODA
  stretch training.

**Train time:** stretch **actions only**; observation windows are indexed separately in
``OptionLabeledZarrDataset`` (row 11), not inside this class.

References
----------
- ``project_proposal.md`` — Variable-Horizon Action Chunking via Temporal Stretching
- ``project_plan.md`` §2 (duration decode: mean-pool → ``unstretch``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

InterpolationMode = Literal["linear"]


def _linear_resample_time(
    sequence: np.ndarray,
    target_len: int,
) -> np.ndarray:
    """
    Resample ``sequence`` of shape ``(T, D)`` to length ``target_len`` along axis 0.

    Uses ``np.interp`` per feature with uniform source times ``0 .. T-1``.
    For ``T=1``, every output step repeats the single row (degenerate segment).
    """
    seq = np.asarray(sequence, dtype=np.float64)
    if seq.ndim != 2:
        raise ValueError(f"sequence must be 2-D (T, D), got shape {seq.shape}")
    t_in, dim = seq.shape
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")

    if t_in == target_len:
        return seq.astype(np.float32, copy=False)

    if t_in == 0:
        raise ValueError("sequence must have at least one timestep")

    t_src = np.arange(t_in, dtype=np.float64)
    t_dst = np.linspace(0.0, float(t_in - 1), target_len)
    out = np.empty((target_len, dim), dtype=np.float32)
    for d in range(dim):
        out[:, d] = np.interp(t_dst, t_src, seq[:, d])
    return out


@dataclass
class TemporalStretcher:
    """
    Linear temporal resampling between native segment length and fixed ``horizon_stretch_max``.

    Parameters
    ----------
    horizon_stretch_max
        Fixed action sequence length after stretching (training hyperparameter; typically
        ``>=`` the longest option segment in the dataset). Sets the low-level U-Net
        ``horizon`` / action chunk size for SODA — distinct from eval ``action_horizon``.
    interpolation
        Only ``"linear"`` is supported today.
    eps
        Small constant for numeric stability (reserved; linear interp handles ``T=1``).
    """

    horizon_stretch_max: int
    interpolation: InterpolationMode = "linear"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.horizon_stretch_max < 1:
            raise ValueError(
                f"horizon_stretch_max must be >= 1, got {self.horizon_stretch_max}"
            )
        if self.interpolation != "linear":
            raise ValueError(f"unsupported interpolation: {self.interpolation!r}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    @staticmethod
    def normalize_duration(segment_steps: int, horizon_stretch_max: int) -> float:
        """
        Map native segment length to ``(0, 1]`` for the duration channel target.

        Same as ``segment_steps / horizon_stretch_max`` (formerly discussed as
        ``h_i / h_max``).
        """
        if segment_steps < 1:
            raise ValueError(f"segment_steps must be >= 1, got {segment_steps}")
        if horizon_stretch_max < 1:
            raise ValueError(f"horizon_stretch_max must be >= 1, got {horizon_stretch_max}")
        return float(segment_steps) / float(horizon_stretch_max)

    @staticmethod
    def normalize_horizon(h_orig: int, horizon_stretch_max: int) -> float:
        """Alias for :meth:`normalize_duration` (kept for call-site readability)."""
        return TemporalStretcher.normalize_duration(h_orig, horizon_stretch_max)

    def stretch(
        self,
        actions: np.ndarray,
        *,
        segment_steps: int | None = None,
        h_orig: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Resample a variable-length action segment to length ``horizon_stretch_max``.

        Parameters
        ----------
        actions
            Shape ``(segment_steps, D)`` native expert actions for one option segment.
        segment_steps
            Native length; if ``None``, inferred from ``actions.shape[0]``.
        h_orig
            Deprecated alias for ``segment_steps``.
        """
        act = np.asarray(actions, dtype=np.float64)
        if act.ndim != 2:
            raise ValueError(f"actions must be 2-D (T, D), got shape {act.shape}")

        if segment_steps is None:
            segment_steps = h_orig
        if segment_steps is None:
            segment_steps = int(act.shape[0])
        h_i = int(segment_steps)

        if h_i < 1:
            raise ValueError(f"segment_steps must be >= 1, got {h_i}")
        if act.shape[0] != h_i:
            raise ValueError(
                f"actions length {act.shape[0]} does not match segment_steps={h_i}"
            )
        if h_i > self.horizon_stretch_max:
            raise ValueError(
                f"segment length {h_i} exceeds horizon_stretch_max={self.horizon_stretch_max}; "
                "cap segments in the dataset or increase horizon_stretch_max"
            )

        stretched = _linear_resample_time(act, self.horizon_stretch_max)
        duration_normalized = self.normalize_duration(h_i, self.horizon_stretch_max)
        return stretched, duration_normalized

    def stretch_batch(
        self,
        segments: Sequence[np.ndarray],
        *,
        segment_steps: Sequence[int] | None = None,
        h_orig: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Stretch a list of variable-length segments to a common batch tensor.

        Returns
        -------
        stretched
            Shape ``(N, horizon_stretch_max, D)``.
        duration_normalized
            Shape ``(N,)``, values ``segment_steps / horizon_stretch_max``.
        """
        if len(segments) == 0:
            raise ValueError("segments must be non-empty")

        lengths = list(segment_steps) if segment_steps is not None else list(h_orig or [])
        if lengths and len(lengths) != len(segments):
            raise ValueError("segment_steps length must match segments")
        if not lengths:
            lengths = [None] * len(segments)

        stretched_list: list[np.ndarray] = []
        durations: list[float] = []
        for seg, h_i in zip(segments, lengths):
            out, d_norm = self.stretch(seg, segment_steps=h_i)
            stretched_list.append(out)
            durations.append(d_norm)

        return np.stack(stretched_list, axis=0), np.asarray(durations, dtype=np.float32)

    def unstretch(
        self,
        actions: np.ndarray,
        segment_steps_pred: int | float,
        *,
        h_pred: int | float | None = None,
    ) -> np.ndarray:
        """
        Resample a fixed-length chunk back to ``segment_steps_pred`` steps for execution.

        Parameters
        ----------
        actions
            Shape ``(horizon_stretch_max, D)`` motion channels only (no duration column).
        segment_steps_pred
            Predicted native segment length (typically from ``decode_duration``).
        h_pred
            Deprecated alias for ``segment_steps_pred``.
        """
        if h_pred is not None:
            segment_steps_pred = h_pred

        act = np.asarray(actions, dtype=np.float64)
        if act.ndim != 2:
            raise ValueError(
                f"actions must be 2-D (horizon_stretch_max, D), got shape {act.shape}"
            )
        if act.shape[0] != self.horizon_stretch_max:
            raise ValueError(
                f"actions length {act.shape[0]} != horizon_stretch_max {self.horizon_stretch_max}"
            )

        h_out = int(round(float(segment_steps_pred)))
        h_out = int(np.clip(h_out, 1, self.horizon_stretch_max))
        return _linear_resample_time(act, h_out)

    def decode_duration(self, action_chunk: np.ndarray) -> int:
        """
        Decode predicted native segment length from a ``D+1`` action chunk.

        Mean-pools the last (duration) channel, then maps
        ``round(mean * horizon_stretch_max)`` into ``[1, horizon_stretch_max]``.
        """
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.shape[-1] < 2:
            raise ValueError(
                f"action_chunk needs duration channel (D+1), got shape {chunk.shape}"
            )
        d_norm = float(np.mean(chunk[..., -1]))
        steps = int(round(d_norm * self.horizon_stretch_max))
        return int(np.clip(steps, 1, self.horizon_stretch_max))

    def decode_horizon(self, action_chunk: np.ndarray) -> int:
        """Alias for :meth:`decode_duration` (duration channel, not eval ``action_horizon``)."""
        return self.decode_duration(action_chunk)

    def append_duration_channel(
        self,
        actions: np.ndarray,
        duration_normalized: float,
    ) -> np.ndarray:
        """Build ``(horizon_stretch_max, D+1)`` training target by broadcasting the duration scalar."""
        act = np.asarray(actions, dtype=np.float32)
        if act.ndim != 2 or act.shape[0] != self.horizon_stretch_max:
            raise ValueError(
                f"actions must be (horizon_stretch_max, D) with horizon_stretch_max="
                f"{self.horizon_stretch_max}, got {act.shape}"
            )
        d_ch = np.full((self.horizon_stretch_max, 1), duration_normalized, dtype=np.float32)
        return np.concatenate([act, d_ch], axis=-1)

    def append_horizon_channel(
        self,
        actions: np.ndarray,
        horizon_normalized: float,
    ) -> np.ndarray:
        """Alias for :meth:`append_duration_channel`."""
        return self.append_duration_channel(actions, horizon_normalized)

    def stretch_with_duration(
        self,
        actions: np.ndarray,
        *,
        segment_steps: int | None = None,
        h_orig: int | None = None,
    ) -> np.ndarray:
        """Stretch segment and return ``(horizon_stretch_max, D+1)`` with duration channel filled."""
        stretched, d_norm = self.stretch(
            actions, segment_steps=segment_steps, h_orig=h_orig
        )
        return self.append_duration_channel(stretched, d_norm)

    def stretch_with_horizon(
        self,
        actions: np.ndarray,
        *,
        h_orig: int | None = None,
    ) -> np.ndarray:
        """Alias for :meth:`stretch_with_duration`."""
        return self.stretch_with_duration(actions, h_orig=h_orig)
