"""
Push-T evaluation metrics (project_plan §6).

Records max block–goal overlap (%) at multiple step horizons within each episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

# Step indices (1-based episode steps) for prefix max-overlap reporting.
DEFAULT_OVERLAP_CHECKPOINTS: tuple[int, ...] = (150, 200, 250, 300)

# Per-episode success only (not reported in aggregate). Matches diffusion_policy
# PushTEnv: coverage ratio > 0.95 counts as success.
PUSHT_SUCCESS_THRESHOLD = 0.95


@dataclass
class EpisodeMetrics:
    """Metrics from one eval episode."""

    seed: int
    prefix: str  # e.g. "test/" or "train/"
    n_steps: int
    max_overlap_full: float  # max overlap % over full episode (0–100 scale)
    max_overlap_at_step: dict[int, float] = field(default_factory=dict)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["max_overlap_at_step"] = {
            str(k): v for k, v in self.max_overlap_at_step.items()
        }
        return out


def step_rewards_to_overlap_pct(step_rewards: Sequence[float]) -> np.ndarray:
    """Convert per-step reward (coverage ratio in [0,1]) to overlap %."""
    return np.asarray(step_rewards, dtype=np.float64) * 100.0


def compute_episode_metrics(
    step_rewards: Sequence[float],
    *,
    seed: int,
    prefix: str = "test/",
    checkpoints: Sequence[int] = DEFAULT_OVERLAP_CHECKPOINTS,
    success_threshold_pct: float = PUSHT_SUCCESS_THRESHOLD * 100.0,
) -> EpisodeMetrics:
    """
    Compute prefix max overlap at requested step horizons.

    For checkpoint ``t``, uses the first ``t`` environment steps (1-based count)
    and takes ``max(overlap_pct[:t])``.
    """
    overlap = step_rewards_to_overlap_pct(step_rewards)
    n_steps = int(overlap.shape[0])

    max_at: dict[int, float] = {}
    for t in checkpoints:
        end = min(t, n_steps)
        if end <= 0:
            max_at[int(t)] = 0.0
        else:
            max_at[int(t)] = float(np.max(overlap[:end]))

    full_max = float(np.max(overlap)) if n_steps > 0 else 0.0
    return EpisodeMetrics(
        seed=seed,
        prefix=prefix,
        n_steps=n_steps,
        max_overlap_full=full_max,
        max_overlap_at_step=max_at,
        success=full_max >= success_threshold_pct,
    )


def aggregate_episode_metrics(
    episodes: Sequence[EpisodeMetrics],
    *,
    checkpoints: Sequence[int] = DEFAULT_OVERLAP_CHECKPOINTS,
) -> dict[str, Any]:
    """Mean metrics over episodes; also reports DP-style ``test/mean_score`` (0–1)."""
    if not episodes:
        raise ValueError("aggregate_episode_metrics requires at least one episode")

    test_eps = [e for e in episodes if e.prefix.startswith("test")]
    pool = test_eps if test_eps else list(episodes)

    full_maxes = [e.max_overlap_full for e in pool]
    mean_full_pct = float(np.mean(full_maxes))

    out: dict[str, Any] = {
        "n_episodes": len(pool),
        "mean_score": mean_full_pct / 100.0,  # 0–1, DP-compatible
    }
    for t in checkpoints:
        vals = [e.max_overlap_at_step.get(int(t), 0.0) for e in pool]
        out[f"mean_score@{t}"] = float(np.mean(vals)) / 100.0

    # Per-episode detail for debugging / plotting
    out["episodes"] = [e.to_dict() for e in episodes]
    return out
