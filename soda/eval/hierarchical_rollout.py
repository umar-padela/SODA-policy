"""
Full-episode hierarchical rollout (π_high + π_low + β) with annotated video.

Each frame is burned with: frame index (red), skill border + label, β value,
duration channel, chunk cursor, and REPLAN indicator when diffusion re-runs.

Used by modal_rollout_hierarchical.py to produce downloadable annotated MP4s.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from soda.eval.metrics import DEFAULT_OVERLAP_CHECKPOINTS, compute_episode_metrics
from soda.eval.policy_loaders import (
    _require_diffusion_policy,
    load_soda_policy_and_cfg,
)
from soda.eval.pusht_rollout import _obs_to_torch
from soda.eval.segment_rollout import capture_sim_rgb, make_pusht_env_for_segment
from soda.inference.hierarchical_controller import HierarchicalPolicy
from soda.option_discovery.supervised.pusht.episode_video import compile_frames_to_mp4
from soda.option_discovery.supervised.pusht.frame_overlays import (
    LowPolicyOverlayInfo,
    apply_rollout_frame_overlays,
)


@dataclass
class HierarchicalEpisodeResult:
    episode_idx: int
    seed: int
    n_steps: int
    n_replans: int
    metrics: dict[str, Any]
    option_sequence: list[int]
    video_path: Path | None


def rollout_hierarchical_episode(
    policy: HierarchicalPolicy,
    *,
    seed: int,
    episode_idx: int,
    n_obs_steps: int,
    max_steps: int = 300,
    render_size: int = 96,
    fps: int = 10,
    legacy_test: bool = True,
    output_dir: Path | None = None,
    record_video: bool = True,
    video_overlap_threshold: float | None = None,
) -> HierarchicalEpisodeResult:
    """
    Roll π_high + π_low for one seeded episode; return metrics + annotated MP4.

    Uses n_action_steps=1 wrapper so a frame is captured at every sim step.
    """
    _require_diffusion_policy()

    env = make_pusht_env_for_segment(
        n_obs_steps=n_obs_steps,
        n_action_steps=1,
        max_episode_steps=max_steps,
        legacy=legacy_test,
        render_size=render_size,
    )
    env.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    obs = env.reset()
    policy.reset()

    frames_raw: list[np.ndarray] = [capture_sim_rgb(env)]
    overlays: list[LowPolicyOverlayInfo] = [LowPolicyOverlayInfo()]
    option_ids: list[int] = [0]
    n_replans = 0
    done = False
    n_steps = 0

    while not done and n_steps < max_steps:
        obs_dict = _obs_to_torch(obs, policy)
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        overlay = policy.last_overlay
        opt_id = policy.last_option_id if policy.last_option_id is not None else 0
        if overlay.replan:
            n_replans += 1

        action = action_dict["action"].detach().cpu().numpy()
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]

        obs, _reward, done, _info = env.step(action)
        done = bool(done)
        n_steps += 1
        frames_raw.append(capture_sim_rgb(env))
        overlays.append(overlay)
        option_ids.append(opt_id)

    # Backfill frame-0 option with whatever π_high chose on step 1
    if len(option_ids) > 1:
        option_ids[0] = option_ids[1]
        overlays[0] = overlays[1]

    step_rewards = list(env.reward)
    metrics_obj = compute_episode_metrics(
        step_rewards,
        seed=seed,
        prefix="episode/",
        checkpoints=DEFAULT_OVERLAP_CHECKPOINTS,
    )
    metrics = metrics_obj.to_dict()
    metrics.update(
        {
            "episode_idx": episode_idx,
            "seed": seed,
            "n_steps": n_steps,
            "n_replans": n_replans,
        }
    )

    max_overlap = metrics.get("max_overlap_full", 0.0)
    save_video = record_video and bool(frames_raw)
    if save_video and video_overlap_threshold is not None:
        save_video = max_overlap < video_overlap_threshold

    video_path: Path | None = None
    if save_video:
        annotated = [
            apply_rollout_frame_overlays(
                frame,
                frame_index=i,
                option_id=option_ids[min(i, len(option_ids) - 1)],
                low_policy_info=overlays[min(i, len(overlays) - 1)],
                burn_state=False,
            )
            for i, frame in enumerate(frames_raw)
        ]
        out_dir = output_dir or Path("experiments/pusht/hierarchical_rollout")
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / f"ep{episode_idx:04d}_seed{seed}.mp4"
        # Repeat each frame 3x so a 10-fps sim produces a 30-fps file that plays
        # correctly in all players (mp4v at low fps is misread by many decoders).
        repeat = max(1, round(30 / fps))
        frames_30fps = [f for f in annotated for _ in range(repeat)]
        compile_frames_to_mp4(frames_30fps, fps * repeat, video_path)
        summary_path = out_dir / f"ep{episode_idx:04d}_seed{seed}.json"
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"  Saved: {video_path}")

    print(
        f"  ep={episode_idx} seed={seed} steps={n_steps} replans={n_replans} "
        f"max_overlap={max_overlap:.1f}%"
    )
    return HierarchicalEpisodeResult(
        episode_idx=episode_idx,
        seed=seed,
        n_steps=n_steps,
        n_replans=n_replans,
        metrics=metrics,
        option_sequence=option_ids,
        video_path=video_path,
    )


def run_hierarchical_rollouts(
    *,
    config_path: Path,
    high_checkpoint: Path,
    low_checkpoint: Path,
    device: str = "cuda:0",
    n_episodes: int = 5,
    test_start_seed: int = 100000,
    n_action_steps: int = 8,
    beta_transition: float | None = None,
    open_loop: bool = False,
    duration_termination: bool = False,
    max_steps: int = 300,
    legacy_test: bool = True,
    output_dir: Path | None = None,
    record_video: bool = True,
    video_overlap_threshold: float | None = None,
) -> list[HierarchicalEpisodeResult]:
    """
    Load SODA hierarchical policy and roll out ``n_episodes`` seeded episodes.

    Seeds are ``test_start_seed + episode_idx`` (matches Columbia eval convention).

    ``open_loop=True``: π_high resamples ω after every full native chunk; β disabled.
    ``duration_termination=True``: π_high resamples ω when duration channel predicts
    the current skill is done (predicted native_len < n_action_steps); β disabled.
    """
    from dataclasses import replace as dc_replace

    policy, _settings, cfg = load_soda_policy_and_cfg(
        device=device,
        config_name=config_path.stem,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
    )

    # Override n_action_steps from caller (load_soda_policy_and_cfg uses yaml default)
    policy.config = dc_replace(policy.config, n_action_steps=n_action_steps)
    if open_loop:
        policy.config = dc_replace(policy.config, open_loop=True)
        policy.config = dc_replace(policy.config, beta_transition=beta_transition if beta_transition is not None else 2.0)
    elif duration_termination:
        policy.config = dc_replace(policy.config, duration_termination=True, beta_transition=2.0)
    elif beta_transition is not None:
        policy.config = dc_replace(policy.config, beta_transition=beta_transition)

    n_obs_steps = int(cfg.n_obs_steps)
    er = cfg.task.env_runner
    fps = int(getattr(er, "fps", 10))
    render_size = int(getattr(er, "render_size", 96))

    results: list[HierarchicalEpisodeResult] = []
    for ep_idx in range(n_episodes):
        seed = test_start_seed + ep_idx
        result = rollout_hierarchical_episode(
            policy,
            seed=seed,
            episode_idx=ep_idx,
            n_obs_steps=n_obs_steps,
            max_steps=max_steps,
            render_size=render_size,
            fps=fps,
            legacy_test=legacy_test,
            output_dir=output_dir,
            record_video=record_video,
            video_overlap_threshold=video_overlap_threshold,
        )
        results.append(result)

    return results
