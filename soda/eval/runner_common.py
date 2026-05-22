"""
Shared helpers for DP and SODA Push-T eval runners (``dp_runner`` / ``soda_runner``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from soda.eval.metrics import (
    DEFAULT_OVERLAP_CHECKPOINTS,
    aggregate_episode_metrics,
    compute_episode_metrics,
)
from soda.eval.run_naming import DEFAULT_TEST_START_SEED


def vector_env_reset(env: Any) -> Any:
    """
    Reset Columbia ``AsyncVectorEnv`` compatibly with gym 0.21–0.26.

    gym>=0.23 ``VectorEnv.reset()`` passes ``seed=`` into ``reset_async()``, but
    DP's fork only defines ``reset_async(self)``.
    """
    if hasattr(env, "reset_async") and hasattr(env, "reset_wait"):
        env.reset_async()
        return env.reset_wait()
    return env.reset()


def slice_action_chunk(action: np.ndarray, n_action_steps: int) -> np.ndarray:
    """Keep the first ``n_action_steps`` rows per env (policy may return more)."""
    action = np.asarray(action)
    if action.ndim == 1:
        return action[:n_action_steps]
    return action[:, :n_action_steps]


def resolve_test_start_seed(requested: int, cfg: Any) -> int:
    """Use checkpoint/yaml seeds unless the caller set a non-default seed."""
    if requested != DEFAULT_TEST_START_SEED:
        return requested
    return int(cfg.task.env_runner.test_start_seed)


def serialize_runner_log(runner_log: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe copy of runner output (wandb.Video → path string)."""
    out: dict[str, Any] = {}
    for key, value in runner_log.items():
        if key == "soda_metrics":
            continue
        if hasattr(value, "_path"):
            out[key] = value._path
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def runner_log_to_soda_metrics(
    runner_log: dict[str, Any],
    *,
    all_rewards: list[list[float]],
    env_seeds: list[int],
    env_prefixs: list[str],
    checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS,
) -> dict[str, Any]:
    """Map per-episode reward lists to SODA overlap metrics + video paths."""
    episodes = []
    video_paths: list[str] = []
    for rewards, seed, prefix in zip(all_rewards, env_seeds, env_prefixs):
        ep = compute_episode_metrics(
            rewards,
            seed=seed,
            prefix=prefix,
            checkpoints=checkpoints,
        )
        episodes.append(ep)
        video_key = f"{prefix}sim_video_{seed}"
        path = runner_log.get(video_key)
        if path is not None:
            video_paths.append(str(path))

    summary = aggregate_episode_metrics(episodes, checkpoints=checkpoints)
    summary["episodes"] = [e.to_dict() for e in episodes]
    summary["video_paths"] = video_paths
    return summary
