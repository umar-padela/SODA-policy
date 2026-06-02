"""
Push-T eval for trained SODA (π_high + π_low + β).

Mirrors ``soda.eval.dp_runner``: same Columbia env stack (``PushTImageRunner``),
MP4 recording, and overlap metrics. Uses **receding-horizon control** — execute
``n_action_steps`` (default 8) from each cached diffusion plan before replanning,
matching vanilla DP eval. β is still checked every sim step.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any

import numpy as np
import tqdm

from soda.eval.metrics import DEFAULT_OVERLAP_CHECKPOINTS
from soda.eval.policy_loaders import (
    _require_diffusion_policy,
    load_soda_policy_and_cfg,
)
from soda.eval.runner_common import (
    resolve_test_start_seed,
    runner_log_to_soda_metrics,
    serialize_runner_log,
    slice_action_chunk,
    vector_env_reset,
)
from soda.eval.run_naming import DEFAULT_TEST_START_SEED


def build_soda_pusht_runner(
    cfg: Any,
    output_dir: Path,
    *,
    n_test: int,
    n_train: int = 0,
    n_test_vis: int,
    n_train_vis: int = 0,
    n_action_steps: int,
    max_steps: int,
    test_start_seed: int,
    train_start_seed: int = 0,
    overlap_checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS,
    noise_eta: float = 0.0,
    noise_rho: float = 0.9,
    noise_scale_by_magnitude: bool = False,
) -> SodaHierarchicalRunner:
    """Construct ``SodaHierarchicalRunner`` (parallel to ``build_pusht_image_runner``)."""
    er = cfg.task.env_runner
    n_envs = max(1, n_train + n_test)
    return SodaHierarchicalRunner(
        overlap_checkpoints=overlap_checkpoints,
        output_dir=str(output_dir),
        n_train=n_train,
        n_train_vis=n_train_vis,
        n_test=n_test,
        n_test_vis=n_test_vis,
        train_start_seed=train_start_seed,
        legacy_test=bool(er.legacy_test),
        test_start_seed=test_start_seed,
        max_steps=max_steps,
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(n_action_steps),
        fps=int(getattr(er, "fps", 10)),
        crf=int(getattr(er, "crf", 22)),
        render_size=int(getattr(er, "render_size", 96)),
        past_action=bool(getattr(cfg, "past_action_visible", False)),
        tqdm_interval_sec=float(getattr(er, "tqdm_interval_sec", 5.0)),
        n_envs=n_envs,
        noise_eta=noise_eta,
        noise_rho=noise_rho,
        noise_scale_by_magnitude=noise_scale_by_magnitude,
    )


class SodaHierarchicalRunner:
    """
    Thin wrapper around DP ``PushTImageRunner`` env/video stack.

    Calls ``HierarchicalPolicy.predict_action`` once per sim step; the controller
    caches diffusion plans and replans every ``n_action_steps`` (unless β fires).
    """

    def __init__(
        self,
        overlap_checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS,
        noise_eta: float = 0.0,
        noise_rho: float = 0.9,
        noise_scale_by_magnitude: bool = False,
        **kwargs: Any,
    ):
        _require_diffusion_policy()
        from soda.eval.pusht_image_runner import PushTImageRunner

        self._overlap_checkpoints = overlap_checkpoints
        self._noise_eta = float(noise_eta)
        self._noise_rho = float(noise_rho)
        self._noise_scale = bool(noise_scale_by_magnitude)
        self._inner = PushTImageRunner(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def run(self, policy: Any) -> dict[str, Any]:
        """Same loop as ``SodaPushTImageRunner`` / DP runner."""
        import torch

        from diffusion_policy.common.pytorch_util import dict_apply
        from soda.eval.action_noise import TemporallyCorrelatedNoise

        device = policy.device
        env = self._inner.env

        n_envs = len(self._inner.env_fns)
        n_inits = len(self._inner.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths: list[Any] = [None] * n_inits
        all_rewards: list[list[float]] = [None] * n_inits  # type: ignore[list-item]

        noise = TemporallyCorrelatedNoise(
            eta=self._noise_eta, rho=self._noise_rho, n_envs=n_envs, action_dim=2
        )

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_local_slice = slice(0, end - start)

            this_init_fns = self._inner.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self._inner.env_init_fn_dills[0]] * n_diff)

            env.call_each(
                "run_dill_function",
                args_list=[(x,) for x in this_init_fns],
            )

            obs = vector_env_reset(env)
            policy.reset()
            noise.reset()

            pbar = tqdm.tqdm(
                total=self._inner.max_steps,
                desc=f"Eval SodaHierarchicalRunner {chunk_idx + 1}/{n_chunks}",
                leave=False,
                mininterval=self._inner.tqdm_interval_sec,
            )
            done = False
            while not done:
                np_obs_dict = dict(obs)
                if self._inner.past_action:
                    raise NotImplementedError("past_action eval path not wired yet")

                obs_dict = dict_apply(
                    np_obs_dict,
                    lambda x: torch.from_numpy(x).to(device=device),
                )

                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                np_action_dict = dict_apply(
                    action_dict,
                    lambda x: x.detach().to("cpu").numpy(),
                )
                action = slice_action_chunk(np_action_dict["action"], 1)
                action = noise.apply_to_chunk(action, scale_by_magnitude=self._noise_scale)

                obs, _reward, done, _info = env.step(action)
                done = bool(np.all(done))
                pbar.update(action.shape[1] if action.ndim > 1 else 1)

            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]

        vector_env_reset(env)

        max_rewards = collections.defaultdict(list)
        log_data: dict[str, Any] = {}
        for i in range(n_inits):
            seed = self._inner.env_seeds[i]
            prefix = self._inner.env_prefixs[i]
            max_reward = float(np.max(all_rewards[i]))
            max_rewards[prefix].append(max_reward)
            log_data[f"{prefix}sim_max_reward_{seed}"] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                import wandb.sdk.data_types.video as wv

                log_data[f"{prefix}sim_video_{seed}"] = wv.Video(video_path)

        for prefix, value in max_rewards.items():
            log_data[prefix + "mean_score"] = float(np.mean(value))

        serialized = serialize_runner_log(log_data)
        soda_metrics = runner_log_to_soda_metrics(
            serialized,
            all_rewards=all_rewards,
            env_seeds=self._inner.env_seeds,
            env_prefixs=self._inner.env_prefixs,
            checkpoints=self._overlap_checkpoints,
        )
        log_data["soda_metrics"] = soda_metrics
        return log_data


def run_soda_pusht_eval(
    *,
    checkpoint: Path,
    output_dir: Path,
    device: str = "cuda:0",
    config_name: str = "soda_supervised",
    n_test: int = 5,
    n_train: int = 0,
    n_test_vis: int | None = None,
    n_train_vis: int = 0,
    n_action_steps: int = 8,
    max_steps: int = 300,
    test_start_seed: int = DEFAULT_TEST_START_SEED,
    train_start_seed: int = 0,
    record_video: bool = True,
    overlap_checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS,
    high_checkpoint: Path | None = None,
    low_checkpoint: Path | None = None,
    noise_eta: float = 0.0,
    noise_rho: float = 0.9,
    noise_scale_by_magnitude: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load SODA checkpoint + yaml, run hierarchical policy in DP env runner.

    ``n_action_steps`` is the eval receding-horizon window (default 8): replan
    diffusion at most every ``n_action_steps`` sim steps unless β fires earlier.

    Returns ``(soda_metrics, runner_log)`` — same contract as ``run_dp_pusht_eval``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "media").mkdir(parents=True, exist_ok=True)

    policy, _settings, cfg = load_soda_policy_and_cfg(
        checkpoint,
        device=device,
        config_name=config_name,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
    )
    seed = resolve_test_start_seed(test_start_seed, cfg)

    if n_test_vis is None:
        n_test_vis = min(n_test, 4) if record_video else 0
    if not record_video:
        n_test_vis = 0

    runner = build_soda_pusht_runner(
        cfg,
        output_dir,
        n_test=n_test,
        n_train=n_train,
        n_test_vis=n_test_vis,
        n_train_vis=n_train_vis,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        test_start_seed=seed,
        train_start_seed=train_start_seed,
        overlap_checkpoints=overlap_checkpoints,
        noise_eta=noise_eta,
        noise_rho=noise_rho,
        noise_scale_by_magnitude=noise_scale_by_magnitude,
    )

    runner_log = runner.run(policy)
    soda_metrics = runner_log.pop("soda_metrics")
    return soda_metrics, serialize_runner_log(runner_log)
