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


def get_action_normalizer_scale(policy: Any) -> np.ndarray | None:
    """
    Return per-dim scale factor so that noise with η=1.0 equals 1 normalised
    action unit (roughly the half-range of the training data).

    For limits-normalised actions, normalizer.scale = 2 / (max - min), so
    the inverse gives (max - min) / 2 — the pixel-space equivalent of 1
    normalised unit.  Returns None if the normalizer cannot be found.
    """
    for source in [policy, getattr(policy, "low_policy", None)]:
        if source is None:
            continue
        norm = getattr(source, "normalizer", None)
        if norm is None:
            continue
        try:
            scale_param = norm["action"].params_dict["scale"].detach().cpu().numpy()
            return (1.0 / scale_param).astype(np.float32)
        except Exception:
            continue
    return None


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
    if action.ndim == 2:
        return action[:n_action_steps]
    return action[:, :n_action_steps]


def _tensor_to_numpy(action: Any) -> np.ndarray:
    if hasattr(action, "detach"):
        return action.detach().cpu().numpy()
    return np.asarray(action)


def extract_env_action_chunk(
    action_dict: dict[str, Any],
    n_action_steps: int,
    *,
    env_action_dim: int = 2,
    squeeze_batch: bool = False,
) -> np.ndarray:
    """
    Format policy output for ``MultiStepWrapper`` / Push-T env.

    π_low: prefer ``action_unstretched`` (duration-decoded native time), then take
    the first ``n_action_steps`` motion rows. Vanilla DP: use ``action`` directly.
    """
    if "action_unstretched" in action_dict and action_dict["action_unstretched"] is not None:
        action = _tensor_to_numpy(action_dict["action_unstretched"])
    else:
        action = _tensor_to_numpy(action_dict["action"])
        if action.shape[-1] > env_action_dim:
            action = action[..., :env_action_dim]

    if squeeze_batch and action.ndim == 3 and action.shape[0] == 1:
        action = action[0]
    return slice_action_chunk(action, n_action_steps)


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
