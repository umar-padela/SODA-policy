"""
Expert-anchored π_low rollouts on Push-T.

Reset sim to the zarr state at an option segment start, roll out π_low with a
fixed skill id, and optionally export a side-by-side expert vs policy MP4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from soda.dataset.option_aware_dataset import OptionSegment
from soda.eval.low_only_runner import wrap_low_policy_for_rollout
from soda.eval.metrics import DEFAULT_OVERLAP_CHECKPOINTS, compute_episode_metrics
from soda.eval.policy_loaders import load_low_policy_from_checkpoint, load_pusht_yaml
from soda.eval.pusht_rollout import _actions_for_env, _obs_to_torch
from soda.eval.segment_zarr import (
    REPRESENTATIVE_MIN_SEGMENT_LENGTH,
    ZarrSegmentStore,
    resolve_segment,
    select_representative_segments,
)
from soda.option_discovery.supervised.pusht.episode_video import compile_frames_to_mp4
from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES


@dataclass(frozen=True)
class SegmentRolloutConfig:
    checkpoint: Path
    config_path: Path
    zarr_path: Path
    label_key: str = "option_id_supervised"
    device: str = "cuda:0"
    fixed_option_id: int | None = None  # default: segment.option_id
    n_obs_steps: int = 2
    n_action_steps: int = 8
    max_steps: int | None = None  # default: remaining segment length
    legacy_test: bool = True
    render_size: int = 96
    fps: int = 10
    record_video: bool = True
    output_dir: Path | None = None


@dataclass
class SegmentRolloutResult:
    segment: OptionSegment
    segment_index: int
    anchor_frame: int
    fixed_option_id: int
    n_policy_steps: int
    metrics: dict[str, Any]
    video_path: Path | None
    policy_frames: list[np.ndarray]
    expert_frames: list[np.ndarray]
    step_rewards: list[float]


def _inner_pusht_env(env: Any) -> Any:
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur


def make_pusht_env_for_segment(
    *,
    n_obs_steps: int,
    n_action_steps: int,
    max_episode_steps: int,
    legacy: bool = True,
    render_size: int = 96,
) -> Any:
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

    inner = PushTImageEnv(legacy=legacy, render_size=render_size)
    return MultiStepWrapper(
        inner,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
    )


def reset_env_to_zarr_state(env: Any, state_vector: np.ndarray) -> Any:
    """Reset wrapped Push-T env so physics matches zarr ``state`` at segment start."""
    inner = _inner_pusht_env(env)
    inner.reset_to_state = np.asarray(state_vector, dtype=np.float64).reshape(5)
    inner._seed = 0
    return env.reset()


def capture_sim_rgb(env: Any) -> np.ndarray:
    """Current sim frame ``(H, W, 3)`` uint8 RGB."""
    inner = _inner_pusht_env(env)
    return np.asarray(inner.render(mode="rgb_array"), dtype=np.uint8)


def compose_side_by_side_frames(
    expert_frames: list[np.ndarray],
    policy_frames: list[np.ndarray],
    *,
    expert_label: str = "expert",
    policy_label: str = "policy",
) -> list[np.ndarray]:
    """Pair expert zarr frames with policy sim frames.

    The longer stream sets video length; the shorter stream holds its last frame.
    Pass the full expert segment on the left and the (possibly shorter) policy
    rollout on the right so early env termination does not truncate the expert.
    """
    if not expert_frames and not policy_frames:
        raise ValueError("need at least one frame stream")
    n = max(len(expert_frames), len(policy_frames))
    out: list[np.ndarray] = []
    for i in range(n):
        exp = expert_frames[min(i, len(expert_frames) - 1)] if expert_frames else policy_frames[0]
        pol = policy_frames[min(i, len(policy_frames) - 1)] if policy_frames else expert_frames[0]
        if exp.shape[:2] != pol.shape[:2]:
            pol = cv2.resize(pol, (exp.shape[1], exp.shape[0]), interpolation=cv2.INTER_AREA)
        combined = np.concatenate([exp, pol], axis=1)
        display = combined.copy()
        cv2.putText(
            display,
            expert_label,
            (4, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            policy_label,
            (exp.shape[1] + 4, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"t={i}",
            (4, display.shape[0] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        out.append(display)
    return out


def rollout_segment(
    cfg: SegmentRolloutConfig,
    store: ZarrSegmentStore,
    segment: OptionSegment,
    *,
    segment_index: int,
    anchor_frame: int | None = None,
) -> SegmentRolloutResult:
    """
    Roll π_low from expert segment start state; compare to expert zarr frames.

    ``anchor_frame`` defaults to ``segment.start`` (segment-first anchor, same as val).
    """
    import torch

    anchor = int(segment.start if anchor_frame is None else anchor_frame)
    if anchor < segment.start or anchor >= segment.end:
        raise ValueError(
            f"anchor {anchor} must lie in [{segment.start}, {segment.end})"
        )

    option_id = (
        int(segment.option_id)
        if cfg.fixed_option_id is None
        else int(cfg.fixed_option_id)
    )
    remaining = int(segment.end - anchor)
    step_budget = remaining if cfg.max_steps is None else min(int(cfg.max_steps), remaining)

    eval_cfg = load_pusht_yaml(cfg.config_path)
    from soda.eval.policy_loaders import _cfg_get

    n_obs_steps = int(_cfg_get(eval_cfg, "n_obs_steps", cfg.n_obs_steps))
    policy = load_low_policy_from_checkpoint(
        cfg.checkpoint,
        device=cfg.device,
        eval_cfg=eval_cfg,
    )
    policy.eval()
    wrapped = wrap_low_policy_for_rollout(
        policy,
        option_id,
        n_action_steps=cfg.n_action_steps,
    )

    env = make_pusht_env_for_segment(
        n_obs_steps=n_obs_steps,
        n_action_steps=cfg.n_action_steps,
        max_episode_steps=step_budget,
        legacy=cfg.legacy_test,
        render_size=cfg.render_size,
    )

    sim_state = store.pusht_env_state(anchor)
    obs = reset_env_to_zarr_state(env, sim_state)
    wrapped.reset()

    policy_frames: list[np.ndarray] = [capture_sim_rgb(env)]
    done = False

    while not done:
        obs_dict = _obs_to_torch(obs, wrapped)
        with torch.no_grad():
            action_dict = wrapped.predict_action(obs_dict)
        action = _actions_for_env(action_dict, cfg.n_action_steps)
        obs, _reward, done, _info = env.step(action)
        done = bool(done)
        policy_frames.append(capture_sim_rgb(env))

    step_rewards = list(env.reward)
    n_policy_steps = len(step_rewards)
    # Full expert segment from anchor (video length follows expert, not policy).
    expert_frames = store.expert_rgb_window(anchor, remaining)
    policy_stopped_early = n_policy_steps < step_budget
    metrics_obj = compute_episode_metrics(
        step_rewards,
        seed=segment_index,
        prefix="segment/",
        checkpoints=DEFAULT_OVERLAP_CHECKPOINTS,
    )
    metrics = metrics_obj.to_dict()
    metrics.update(
        {
            "segment_index": segment_index,
            "episode_idx": segment.episode_idx,
            "option_id": segment.option_id,
            "skill": SKILL_NAMES.get(segment.option_id, str(segment.option_id)),
            "fixed_option_id": option_id,
            "anchor_frame": anchor,
            "segment_start": segment.start,
            "segment_end": segment.end,
            "segment_length": segment.length,
            "step_budget": step_budget,
            "n_policy_steps": n_policy_steps,
            "n_expert_video_frames": len(expert_frames),
            "n_policy_video_frames": len(policy_frames),
            "policy_stopped_early": policy_stopped_early,
        }
    )

    video_path: Path | None = None
    if cfg.record_video:
        out_dir = cfg.output_dir or Path("experiments/segment_rollout")
        out_dir.mkdir(parents=True, exist_ok=True)
        skill = SKILL_NAMES.get(option_id, str(option_id))
        stem = (
            f"seg{segment_index:04d}_ep{segment.episode_idx:03d}_"
            f"o{option_id}_{skill.lower()}_anchor{anchor}"
        )
        video_path = out_dir / f"{stem}.mp4"
        side_by_side = compose_side_by_side_frames(expert_frames, policy_frames)
        compile_frames_to_mp4(side_by_side, cfg.fps, video_path)

        summary_path = out_dir / f"{stem}.json"
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return SegmentRolloutResult(
        segment=segment,
        segment_index=segment_index,
        anchor_frame=anchor,
        fixed_option_id=option_id,
        n_policy_steps=n_policy_steps,
        metrics=metrics,
        video_path=video_path,
        policy_frames=policy_frames,
        expert_frames=expert_frames,
        step_rewards=step_rewards,
    )


def run_segment_rollout_from_cli(
    *,
    checkpoint: Path,
    config_path: Path,
    zarr_path: Path,
    segment_index: int | None = None,
    episode_idx: int | None = None,
    option_id: int | None = None,
    segment_rank: int = 0,
    fixed_option_id: int | None = None,
    label_key: str = "option_id_supervised",
    device: str = "cuda:0",
    n_action_steps: int = 8,
    max_steps: int | None = None,
    output_dir: Path | None = None,
    record_video: bool = True,
    no_video: bool = False,
) -> SegmentRolloutResult:
    store = ZarrSegmentStore.open(zarr_path, label_key=label_key)
    segment, seg_idx = resolve_segment(
        store,
        segment_index=segment_index,
        episode_idx=episode_idx,
        option_id=option_id,
        segment_rank=segment_rank,
    )
    cfg = SegmentRolloutConfig(
        checkpoint=Path(checkpoint),
        config_path=Path(config_path),
        zarr_path=Path(zarr_path),
        label_key=label_key,
        device=device,
        fixed_option_id=fixed_option_id,
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        output_dir=output_dir,
        record_video=record_video and not no_video,
    )
    return rollout_segment(cfg, store, segment, segment_index=seg_idx)


def run_representative_segment_rollouts(
    *,
    checkpoint: Path,
    config_path: Path,
    zarr_path: Path,
    per_skill: int = 2,
    min_segment_length: int = REPRESENTATIVE_MIN_SEGMENT_LENGTH,
    fixed_option_id: int | None = None,
    label_key: str = "option_id_supervised",
    device: str = "cuda:0",
    n_action_steps: int = 8,
    max_steps: int | None = None,
    output_dir: Path | None = None,
    record_video: bool = True,
    no_video: bool = False,
) -> list[SegmentRolloutResult]:
    """Roll out ``per_skill`` median-length segments for each skill (default 6 total)."""
    store = ZarrSegmentStore.open(zarr_path, label_key=label_key)
    picks = select_representative_segments(
        store,
        per_skill=per_skill,
        min_length=min_segment_length,
    )
    results: list[SegmentRolloutResult] = []
    for seg_idx, _segment in picks:
        results.append(
            run_segment_rollout_from_cli(
                checkpoint=checkpoint,
                config_path=config_path,
                zarr_path=zarr_path,
                segment_index=seg_idx,
                fixed_option_id=fixed_option_id,
                label_key=label_key,
                device=device,
                n_action_steps=n_action_steps,
                max_steps=max_steps,
                output_dir=output_dir,
                record_video=record_video,
                no_video=no_video,
            )
        )
    return results
