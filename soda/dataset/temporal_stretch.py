"""
Variable-horizon resampling for SODA low-level training (project_plan §2, §7 row 10).

Maps option segments of native length ``segment_steps`` to a fixed training length
``horizon`` via linear interpolation, and reverses the mapping at inference using
the predicted **duration channel** (mean-pool → ``unstretch``).

Naming (avoid confusion with eval control)
-----------------------------------------
- ``horizon``: U-Net action prediction length (DP convention). For SODA, variable
  skills are stretched to this fixed length at train time. ``null`` in yaml → longest
  option segment in the dataset. **Not** the sim step budget.
- ``n_action_steps``: eval — how many actions from ``predict_action`` to execute
  before replanning (``soda/eval/*_runner.py``; P0 default ``n_action_steps=8``).

**Train time:** stretch **actions only**; observation windows are indexed separately in
``OptionAwareDataset`` (row 11), not inside this class.

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


def _check_horizon(horizon: int) -> int:
    h = int(horizon)
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    return h


def segment_steps_from_normalized_duration(
    d_norm: float | np.floating,
    horizon: int,
) -> int:
    """``round(d_norm * horizon)`` clipped to ``[1, horizon]``."""
    h = _check_horizon(horizon)
    steps = int(round(float(d_norm) * h))
    return int(np.clip(steps, 1, h))


def segment_steps_from_normalized_duration_batch(
    d_norm: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Vectorized :func:`segment_steps_from_normalized_duration`."""
    h = _check_horizon(horizon)
    arr = np.asarray(d_norm, dtype=np.float64)
    steps = np.round(arr * h).astype(np.int64)
    return np.clip(steps, 1, h)


def _mean_duration_channel(action_chunk: np.ndarray) -> float | np.ndarray:
    chunk = np.asarray(action_chunk, dtype=np.float64)
    if chunk.shape[-1] < 2:
        raise ValueError(
            f"action_chunk needs duration channel (D+1), got shape {chunk.shape}"
        )
    duration = chunk[..., -1]
    if duration.ndim == 1:
        return float(np.mean(duration))
    if duration.ndim == 2:
        return np.mean(duration, axis=-1)
    raise ValueError(
        f"action_chunk must be (T, D+1) or (B, T, D+1), got shape {chunk.shape}"
    )


def decode_segment_steps(
    action_chunk: np.ndarray,
    horizon: int,
) -> int | np.ndarray:
    """
    Mean-pool duration channel → predicted native segment length (project_plan §2).

    ``(T, D+1)`` → ``int``; ``(B, T, D+1)`` → ``(B,)`` int array.
    """
    d_norm = _mean_duration_channel(action_chunk)
    if isinstance(d_norm, float):
        return segment_steps_from_normalized_duration(d_norm, horizon)
    return segment_steps_from_normalized_duration_batch(d_norm, horizon)


@dataclass
class TemporalStretcher:
    """
    Linear temporal resampling between native segment length and fixed ``horizon``.

    Parameters
    ----------
    horizon
        Fixed action sequence length after stretching (typically ``>=`` the longest
        option segment in the dataset). Sets the low-level U-Net prediction length —
        distinct from eval ``n_action_steps``.
    interpolation
        Only ``"linear"`` is supported today.
    eps
        Small constant for numeric stability (reserved; linear interp handles ``T=1``).
    """

    horizon: int
    interpolation: InterpolationMode = "linear"
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.interpolation != "linear":
            raise ValueError(f"unsupported interpolation: {self.interpolation!r}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

    @staticmethod
    def normalize_duration(segment_steps: int, horizon: int) -> float:
        """
        Map native segment length to ``(0, 1]`` for the duration channel target.

        Same as ``segment_steps / horizon``.
        """
        if segment_steps < 1:
            raise ValueError(f"segment_steps must be >= 1, got {segment_steps}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        return float(segment_steps) / float(horizon)

    @staticmethod
    def normalize_horizon(h_orig: int, horizon: int) -> float:
        """Alias for :meth:`normalize_duration` (kept for call-site readability)."""
        return TemporalStretcher.normalize_duration(h_orig, horizon)

    def stretch(
        self,
        actions: np.ndarray,
        *,
        segment_steps: int | None = None,
        h_orig: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Resample a variable-length action segment to length ``horizon``.

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
        if h_i > self.horizon:
            raise ValueError(
                f"segment length {h_i} exceeds horizon={self.horizon}; "
                "cap segments in the dataset or increase horizon"
            )

        stretched = _linear_resample_time(act, self.horizon)
        duration_normalized = self.normalize_duration(h_i, self.horizon)
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
            Shape ``(N, horizon, D)``.
        duration_normalized
            Shape ``(N,)``, values ``segment_steps / horizon``.
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
            Shape ``(horizon, D)`` motion channels only (no duration column).
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
                f"actions must be 2-D (horizon, D), got shape {act.shape}"
            )
        if act.shape[0] != self.horizon:
            raise ValueError(
                f"actions length {act.shape[0]} != horizon {self.horizon}"
            )

        h_out = int(round(float(segment_steps_pred)))
        h_out = int(np.clip(h_out, 1, self.horizon))
        return _linear_resample_time(act, h_out)

    def decode_duration(self, action_chunk: np.ndarray) -> int:
        """Decode native segment length from a ``D+1`` chunk (delegates to :func:`decode_segment_steps`)."""
        steps = decode_segment_steps(action_chunk, self.horizon)
        if not isinstance(steps, (int, np.integer)):
            raise ValueError("decode_duration expects one chunk (T, D+1), not a batch")
        return int(steps)

    def decode_horizon(self, action_chunk: np.ndarray) -> int:
        """Alias for :meth:`decode_duration` (duration channel, not eval ``n_action_steps``)."""
        return self.decode_duration(action_chunk)

    def append_duration_channel(
        self,
        actions: np.ndarray,
        duration_normalized: float,
    ) -> np.ndarray:
        """Build ``(horizon, D+1)`` training target by broadcasting the duration scalar."""
        act = np.asarray(actions, dtype=np.float32)
        if act.ndim != 2 or act.shape[0] != self.horizon:
            raise ValueError(
                f"actions must be (horizon, D) with horizon={self.horizon}, got {act.shape}"
            )
        d_ch = np.full((self.horizon, 1), duration_normalized, dtype=np.float32)
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
        """Stretch segment and return ``(horizon, D+1)`` with duration channel filled."""
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
