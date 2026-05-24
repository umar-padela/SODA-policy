"""
Legacy Push-T rollouts (custom loop). Prefer ``dp_runner`` / ``soda_runner``.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import numpy as np

from soda.eval.metrics import (
    DEFAULT_OVERLAP_CHECKPOINTS,
    EpisodeMetrics,
    aggregate_episode_metrics,
)
from soda.eval.policy_loaders import DPEvalSettings, PushTPolicy
from soda.eval.runner_common import extract_env_action_chunk

ProgressMode = Literal["none", "log", "tqdm"]


def _resolve_progress_mode(show_progress: bool) -> ProgressMode:
    """
    Default: plain ``print`` lines (show up in Modal run logs and local terminal).

    Optional: set env ``SODA_EVAL_TQDM=1`` for ``tqdm`` bars when stderr is a TTY.
    """
    if not show_progress:
        return "none"
    if os.environ.get("SODA_EVAL_TQDM", "").strip().lower() in ("1", "true", "yes"):
        try:
            import sys

            if sys.stderr.isatty():
                return "tqdm"
        except (AttributeError, OSError):
            pass
    return "log"


def _step_log_interval(step_budget: int) -> int:
    """Log roughly every 5% of an episode, but every step when the episode is short."""
    if step_budget <= 30:
        return 1
    return max(1, step_budget // 20)


def _log_step_progress(
    label: str,
    n_steps: int,
    step_budget: int,
    reward: float | None,
) -> None:
    pct = 100.0 * n_steps / step_budget
    rew = f"{reward:.3f}" if reward is not None else "-"
    print(
        f"[pusht eval] {label}: step {n_steps}/{step_budget} ({pct:.0f}%) "
        f"reward={rew}",
        flush=True,
    )


def _require_diffusion_policy() -> None:
    from soda.eval.policy_loaders import DP_ROOT

    if not (DP_ROOT / "eval.py").is_file():
        raise FileNotFoundError(
            "third_party/diffusion_policy missing — init submodule first."
        )


def make_pusht_eval_env(
    settings: DPEvalSettings,
    *,
    seed: int,
    n_action_steps: int,
    legacy_test: bool | None = None,
    max_steps: int | None = None,
):
    """Build Push-T image env stack matching DP training (without video recording)."""
    _require_diffusion_policy()
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    legacy = settings.legacy_test if legacy_test is None else legacy_test
    horizon = max_steps if max_steps is not None else settings.max_steps

    env = MultiStepWrapper(
        PushTImageEnv(legacy=legacy, render_size=96),
        n_obs_steps=settings.n_obs_steps,
        n_action_steps=int(n_action_steps),
        max_episode_steps=horizon,
    )
    env.seed(seed)
    return env


def _obs_to_torch(obs: dict, policy: PushTPolicy) -> dict:
    """
    Convert MultiStepWrapper obs to DP policy layout.

    ``BaseImagePolicy.predict_action`` expects each tensor as ``(B, To, …)``.
    Vectorized DP eval uses ``AsyncVectorEnv`` (B = n_envs). Single-env
    rollouts add ``B=1`` here; without it, ``(To, C, H, W)`` is misread as
    ``(B, To, H, W)`` and the vision encoder sees the wrong rank/shape.
    """
    import torch

    device = policy.device
    out: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        t = torch.from_numpy(np.asarray(value)).to(
            device=device, dtype=torch.float32
        )
        out[key] = t.unsqueeze(0)
    return out


def _actions_for_env(
    action_dict: dict[str, Any],
    n_action_steps: int,
    *,
    env_action_dim: int = 2,
) -> np.ndarray:
    """Slice receding window from native (π_low) or DP ``action`` for the sim env."""
    return extract_env_action_chunk(
        action_dict,
        n_action_steps,
        env_action_dim=env_action_dim,
        squeeze_batch=True,
    )


def rollout_episode(
    policy: PushTPolicy,
    env,
    *,
    seed: int,
    n_action_steps: int,
    prefix: str = "test/",
    checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS,
    max_steps: int | None = None,
    show_progress: bool = False,
    episode_label: str = "",
) -> EpisodeMetrics:
    """
    Run one episode and return overlap metrics at step checkpoints.

    Replan after each ``predict_action`` chunk (``n_action_steps=1`` = closed-loop).
    """
    from soda.eval.metrics import compute_episode_metrics

    env.seed(seed)
    obs = env.reset()
    policy.reset()

    done = False
    step_budget = max_steps or getattr(env, "max_episode_steps", None) or 300
    desc = episode_label or f"{prefix}seed={seed}"
    progress_mode = _resolve_progress_mode(show_progress)
    log_interval = _step_log_interval(step_budget)
    last_logged_step = 0

    def _sim_step_count() -> int:
        """Underlying env steps (``MultiStepWrapper.reward`` length)."""
        return len(env.reward)

    step_pbar = None
    if progress_mode == "tqdm":
        from tqdm import tqdm

        step_pbar = tqdm(
            total=step_budget,
            desc=desc,
            leave=False,
            mininterval=0.5,
            dynamic_ncols=False,
        )

    def _sync_step_progress() -> None:
        nonlocal last_logged_step
        n = _sim_step_count()
        if n == 0:
            return
        reward = float(env.reward[-1])

        if step_pbar is not None:
            if n > step_pbar.n:
                step_pbar.update(n - step_pbar.n)
            step_pbar.set_postfix(reward=f"{reward:.3f}", refresh=False)

        if progress_mode == "log":
            if n >= step_budget or n - last_logged_step >= log_interval:
                _log_step_progress(desc, n, step_budget, reward)
                last_logged_step = n

    while not done:
        obs_dict = _obs_to_torch(obs, policy)
        import torch

        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        action = _actions_for_env(action_dict, n_action_steps)
        obs, _reward, done, _info = env.step(action)
        done = bool(done)
        _sync_step_progress()

    if step_pbar is not None:
        step_pbar.close()
    elif progress_mode == "log" and env.reward:
        n = _sim_step_count()
        if n != last_logged_step:
            _log_step_progress(desc, n, step_budget, float(env.reward[-1]))

    step_rewards = list(env.reward)

    return compute_episode_metrics(
        step_rewards,
        seed=seed,
        prefix=prefix,
        checkpoints=checkpoints,
    )


def run_pusht_rollouts(
    policy: PushTPolicy,
    settings: DPEvalSettings,
    *,
    n_test: int = 50,
    test_start_seed: int = 10000,
    n_train: int = 0,
    train_start_seed: int = 0,
    max_steps: int | None = None,
    n_action_steps: int = 1,
    checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Roll out ``policy`` on Push-T and aggregate metrics.

    Default seeds match DP ``PushTImageRunner`` (test from 10000).
    Default ``n_action_steps=1``: replan every sim step.
    """
    episodes: list[EpisodeMetrics] = []
    progress_mode = _resolve_progress_mode(show_progress)

    def _run_split(n: int, start_seed: int, prefix: str) -> None:
        split_name = prefix.strip() or "ep"
        iterator: Any = range(n)
        if progress_mode == "tqdm":
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                desc=f"pusht eval {split_name} (episodes)",
                dynamic_ncols=False,
            )
        for i in iterator:
            seed = start_seed + i
            ep_label = f"{split_name} {i + 1}/{n} seed={seed}"
            if progress_mode == "log":
                print(f"[pusht eval] starting {ep_label}", flush=True)
            env = make_pusht_eval_env(
                settings,
                seed=seed,
                n_action_steps=n_action_steps,
                max_steps=max_steps,
            )
            ep = rollout_episode(
                policy,
                env,
                seed=seed,
                n_action_steps=n_action_steps,
                prefix=prefix,
                checkpoints=checkpoints,
                max_steps=max_steps,
                show_progress=show_progress,
                episode_label=ep_label,
            )
            episodes.append(ep)
            if progress_mode == "log":
                print(
                    f"[pusht eval] finished {ep_label} "
                    f"steps={ep.n_steps} max_overlap={ep.max_overlap_full:.1f}%",
                    flush=True,
                )
            if hasattr(env, "close"):
                env.close()

    _run_split(n_train, train_start_seed, "train/")
    _run_split(n_test, test_start_seed, "test/")

    summary = aggregate_episode_metrics(episodes, checkpoints=checkpoints)
    summary["n_action_steps"] = int(n_action_steps)
    summary["policy_horizon"] = settings.policy_horizon
    summary["max_steps"] = max_steps or settings.max_steps
    summary["n_test"] = n_test
    summary["n_train"] = n_train
    return summary
