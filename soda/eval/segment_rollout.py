"""
Expert-anchored π_low rollouts on Push-T.

Reset sim to the zarr state at an option segment start, roll out π_low with a
fixed skill id via ``FixedOptionLowController`` (same low-level execute path as
hierarchical eval, minus π_high), and export a side-by-side expert | policy MP4
with β / duration overlays on the policy side.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from soda.dataset.option_aware_dataset import OptionSegment
from soda.eval.metrics import DEFAULT_OVERLAP_CHECKPOINTS, compute_episode_metrics
from soda.eval.policy_loaders import (
    load_dp_image_policy_and_cfg,
    load_low_policy_from_checkpoint,
    load_pusht_yaml,
)
from soda.eval.pusht_rollout import _obs_to_torch
from soda.eval.segment_zarr import (
    REPRESENTATIVE_MIN_SEGMENT_LENGTH,
    ZarrSegmentStore,
    resolve_segment,
    select_representative_segments,
)
from soda.experiments.paths import infer_task_slug, segment_rollout_dir
from soda.inference.fixed_option_low_controller import FixedOptionLowController
from soda.inference.hierarchical_controller import HierarchicalPolicyConfig
from soda.option_discovery.supervised.pusht.episode_video import compile_frames_to_mp4
from soda.option_discovery.supervised.pusht.frame_overlays import (
    LowPolicyOverlayInfo,
    apply_rollout_frame_overlays,
)
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
    legacy_test: bool = False  # must be False for zarr-anchored reset: legacy order (pos then angle) shifts block by 45·sin(θ) due to CoG offset
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
            (4, exp.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 200, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            policy_label,
            (exp.shape[1] + 4, pol.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 120, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            f"t={i}",
            (display.shape[1] - 48, display.shape[0] - 6),
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
    # Allow the policy to run for at least 50 frames even if the expert segment is shorter.
    # --max-steps overrides without capping at the expert segment length.
    step_budget = int(cfg.max_steps) if cfg.max_steps is not None else max(remaining, 50)

    eval_cfg = load_pusht_yaml(cfg.config_path)
    from soda.eval.policy_loaders import _cfg_get

    n_obs_steps = int(_cfg_get(eval_cfg, "n_obs_steps", cfg.n_obs_steps))
    n_action_steps = int(_cfg_get(eval_cfg, "n_action_steps", cfg.n_action_steps))
    policy = load_low_policy_from_checkpoint(
        cfg.checkpoint,
        device=cfg.device,
        eval_cfg=eval_cfg,
    )
    policy.eval()

    infer_cfg = _cfg_get(eval_cfg, "inference", {}) or {}
    controller = FixedOptionLowController(
        policy,
        fixed_option_id=option_id,
        config=HierarchicalPolicyConfig(
            n_action_steps=n_action_steps,
            beta_transition=float(_cfg_get(infer_cfg, "beta_transition", 0.9)),
            beta_diffusion_t=int(_cfg_get(infer_cfg, "beta_diffusion_t", 0)),
            env_action_dim=2,
        ),
    )

    # Single sim step per ``predict_action`` (matches ``SodaHierarchicalRunner``).
    env = make_pusht_env_for_segment(
        n_obs_steps=n_obs_steps,
        n_action_steps=1,
        max_episode_steps=step_budget,
        legacy=cfg.legacy_test,
        render_size=cfg.render_size,
    )

    sim_state = store.pusht_env_state(anchor)
    obs = reset_env_to_zarr_state(env, sim_state)
    controller.reset()

    expert_option_ids = np.asarray(
        store.option_ids[int(anchor) : int(anchor + remaining)],
        dtype=np.int32,
    )

    policy_frames_raw: list[np.ndarray] = [capture_sim_rgb(env)]
    policy_overlays: list[LowPolicyOverlayInfo] = [LowPolicyOverlayInfo()]
    beta_trace: list[float] = []
    beta_threshold = float(_cfg_get(infer_cfg, "beta_transition", 0.9))
    beta_fired_in_rollout = False
    done = False
    n_steps = 0
    _BETA_TIMEOUT = step_budget  # max steps before giving up if β never fires

    while not done:
        obs_dict = _obs_to_torch(obs, controller)
        with torch.no_grad():
            action_dict = controller.predict_action(obs_dict)
        overlay = controller.last_overlay
        if overlay.beta is not None:
            beta_trace.append(float(overlay.beta))
        if action_dict.get("beta_fired"):
            beta_fired_in_rollout = True
            policy_frames_raw.append(capture_sim_rgb(env))
            policy_overlays.append(overlay)
            break
        if n_steps >= _BETA_TIMEOUT:
            # β has not fired; cap the rollout to avoid running forever.
            break
        # Use "action" directly — the executor already cursor-tracks the native chunk.
        # extract_env_action_chunk prefers "action_unstretched" (full native chunk) and
        # would always emit native[0] instead of native[cursor].
        action = action_dict["action"].detach().cpu().numpy()
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]  # (1, D)
        obs, _reward, done, _info = env.step(action)
        done = bool(done)
        n_steps += 1
        policy_frames_raw.append(capture_sim_rgb(env))
        policy_overlays.append(controller.last_overlay)

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
            "n_policy_video_frames": len(policy_frames_raw),
            "policy_stopped_early": policy_stopped_early,
            "rollout_mode": "fixed_option_low_controller",
            "beta_threshold": beta_threshold,
            "beta_fired": beta_fired_in_rollout,
            "beta_max": max(beta_trace) if beta_trace else None,
            "beta_min": min(beta_trace) if beta_trace else None,
            "n_beta_samples": len(beta_trace),
        }
    )

    policy_frames = [
        apply_rollout_frame_overlays(
            frame,
            frame_index=i,
            option_id=option_id,
            low_policy_info=policy_overlays[min(i, len(policy_overlays) - 1)],
            burn_state=False,
        )
        for i, frame in enumerate(policy_frames_raw)
    ]
    expert_frames_burned = [
        apply_rollout_frame_overlays(
            frame,
            frame_index=i,
            option_id=int(expert_option_ids[min(i, len(expert_option_ids) - 1)]),
            burn_state=False,
        )
        for i, frame in enumerate(expert_frames)
    ]

    video_path: Path | None = None
    if cfg.record_video:
        task = infer_task_slug(config_path=cfg.config_path)
        out_dir = cfg.output_dir or Path(segment_rollout_dir(task))
        out_dir.mkdir(parents=True, exist_ok=True)
        skill = SKILL_NAMES.get(option_id, str(option_id))
        stem = (
            f"seg{segment_index:04d}_ep{segment.episode_idx:03d}_"
            f"o{option_id}_{skill.lower()}_anchor{anchor}"
        )
        video_path = out_dir / f"{stem}.mp4"
        side_by_side = compose_side_by_side_frames(expert_frames_burned, policy_frames)
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
        policy_frames=policy_frames_raw,
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


# ---------------------------------------------------------------------------
# Vanilla DP segment rollout (comparison baseline)
# ---------------------------------------------------------------------------


def rollout_segment_dp(
    store: ZarrSegmentStore,
    segment: OptionSegment,
    *,
    segment_index: int,
    dp_checkpoint: Path,
    device: str = "cpu",
    n_obs_steps: int = 2,
    n_action_steps: int = 8,
    max_steps: int | None = None,
    render_size: int = 96,
    fps: int = 10,
    record_video: bool = True,
    output_dir: Path | None = None,
    _dp_policy: Any | None = None,
) -> SegmentRolloutResult:
    """
    Roll vanilla DP from the same zarr-anchored start state used for π_low rollouts.

    Resets the sim to the expert segment start, runs the DP policy for
    ``max(remaining_segment_length, 50)`` steps, and saves a side-by-side
    expert | DP video.  No option conditioning or β head — pure diffusion action.

    Pass ``_dp_policy`` to avoid re-loading the checkpoint on every call when
    rolling out multiple segments (see ``run_representative_segment_rollouts_dp``).
    """
    anchor = int(segment.start)
    remaining = int(segment.end - anchor)
    step_budget = int(max_steps) if max_steps is not None else max(remaining, 50)

    if _dp_policy is None:
        dp_policy, _, _ = load_dp_image_policy_and_cfg(dp_checkpoint, device=device)
        dp_policy.eval()
    else:
        dp_policy = _dp_policy

    # Use n_action_steps=1 env so we capture a frame at every sim step.
    # DP replans every n_action_steps via a cursor — same receding-horizon
    # cadence as the real eval but with per-step frame capture for smooth video.
    env = make_pusht_env_for_segment(
        n_obs_steps=n_obs_steps,
        n_action_steps=1,
        max_episode_steps=step_budget,
        legacy=False,
        render_size=render_size,
    )

    sim_state = store.pusht_env_state(anchor)
    obs = reset_env_to_zarr_state(env, sim_state)
    dp_policy.reset()

    policy_frames_raw: list[np.ndarray] = [capture_sim_rgb(env)]
    cached_chunk: np.ndarray | None = None
    cursor = 0
    done = False
    n_sim_steps = 0

    while not done and n_sim_steps < step_budget:
        if cursor == 0 or cached_chunk is None:
            obs_torch = _obs_to_torch(obs, dp_policy)
            with torch.no_grad():
                action_dict = dp_policy.predict_action(obs_torch)
            cached_chunk = action_dict["action"].detach().cpu().numpy()
            if cached_chunk.ndim == 3:
                cached_chunk = cached_chunk[0]  # (n_action_steps, D)
        action = cached_chunk[cursor : cursor + 1]  # (1, D)
        obs, _reward, done, _info = env.step(action)
        done = bool(done)
        n_sim_steps += 1
        cursor = (cursor + 1) % n_action_steps
        policy_frames_raw.append(capture_sim_rgb(env))

    step_rewards = list(env.reward)
    n_policy_steps = len(step_rewards)
    expert_frames = store.expert_rgb_window(anchor, remaining)
    option_id = int(segment.option_id)

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
            "option_id": option_id,
            "skill": SKILL_NAMES.get(option_id, str(option_id)),
            "anchor_frame": anchor,
            "segment_start": segment.start,
            "segment_end": segment.end,
            "segment_length": segment.length,
            "step_budget": step_budget,
            "n_policy_steps": n_policy_steps,
            "n_expert_video_frames": len(expert_frames),
            "n_policy_video_frames": len(policy_frames_raw),
            "policy_stopped_early": n_policy_steps < step_budget,
            "rollout_mode": "dp_vanilla",
            "beta_fired": None,
            "beta_max": None,
            "beta_min": None,
            "n_beta_samples": 0,
        }
    )

    video_path: Path | None = None
    if record_video:
        out_dir = output_dir or Path(segment_rollout_dir("pusht"))
        out_dir.mkdir(parents=True, exist_ok=True)
        skill = SKILL_NAMES.get(option_id, str(option_id))
        stem = (
            f"seg{segment_index:04d}_ep{segment.episode_idx:03d}_"
            f"o{option_id}_{skill.lower()}_anchor{anchor}_dp"
        )
        video_path = out_dir / f"{stem}.mp4"
        side_by_side = compose_side_by_side_frames(
            expert_frames, policy_frames_raw, expert_label="expert", policy_label="dp"
        )
        compile_frames_to_mp4(side_by_side, fps, video_path)

        summary_path = out_dir / f"{stem}.json"
        summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved: {video_path}")

    return SegmentRolloutResult(
        segment=segment,
        segment_index=segment_index,
        anchor_frame=anchor,
        fixed_option_id=option_id,
        n_policy_steps=n_policy_steps,
        metrics=metrics,
        video_path=video_path,
        policy_frames=policy_frames_raw,
        expert_frames=expert_frames,
        step_rewards=step_rewards,
    )


def run_representative_segment_rollouts_dp(
    *,
    dp_checkpoint: Path,
    zarr_path: Path,
    per_skill: int = 2,
    min_segment_length: int = REPRESENTATIVE_MIN_SEGMENT_LENGTH,
    label_key: str = "option_id_supervised",
    device: str = "cpu",
    n_action_steps: int = 8,
    max_steps: int | None = None,
    output_dir: Path | None = None,
    record_video: bool = True,
) -> list[SegmentRolloutResult]:
    """Roll vanilla DP on ``per_skill`` median-length segments per skill (6 total by default)."""
    store = ZarrSegmentStore.open(zarr_path, label_key=label_key)
    picks = select_representative_segments(
        store, per_skill=per_skill, min_length=min_segment_length
    )
    dp_policy, _, _ = load_dp_image_policy_and_cfg(dp_checkpoint, device=device)
    dp_policy.eval()

    results: list[SegmentRolloutResult] = []
    for seg_idx, segment in picks:
        print(
            f"Rolling DP on segment {seg_idx} "
            f"(ep={segment.episode_idx} ω={segment.option_id} len={segment.length})"
        )
        results.append(
            rollout_segment_dp(
                store,
                segment,
                segment_index=seg_idx,
                dp_checkpoint=dp_checkpoint,
                device=device,
                n_action_steps=n_action_steps,
                max_steps=max_steps,
                output_dir=output_dir,
                record_video=record_video,
                _dp_policy=dp_policy,
            )
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Expert-anchored segment rollouts: compare π_low (soda) vs vanilla DP."
    )
    parser.add_argument(
        "--policy",
        choices=("soda", "dp"),
        default="soda",
        help="Policy to roll out: 'soda' = π_low with FixedOptionLowController; 'dp' = vanilla DP",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="π_low checkpoint (soda) or DP checkpoint (dp)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pusht/soda_supervised.yaml"),
        help="SODA yaml config (only needed for --policy soda)",
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        default=Path("data/raw/pusht/pusht.zarr"),
        help="Path to pusht.zarr",
    )
    parser.add_argument(
        "--label-key",
        default="option_id_supervised",
        help="Option label column in zarr",
    )
    parser.add_argument(
        "--representative",
        action="store_true",
        help="Roll out per_skill median-length representative segments per skill",
    )
    parser.add_argument(
        "--per-skill",
        type=int,
        default=2,
        help="Segments per skill when using --representative (default 2 → 6 total for Push-T)",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=None,
        help="Single segment index (alternative to --representative)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for MP4s and JSON summaries (default: experiments/pusht/segment_rollout/)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device (default: cpu for local use)",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Receding-horizon window (default 8; DP replans every N steps)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override step budget per segment (default: max(segment_length, 50))",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 export")
    args = parser.parse_args()

    if not args.representative and args.segment_index is None:
        parser.error("Pass --representative or --segment-index N")

    if args.representative:
        if args.policy == "soda":
            run_representative_segment_rollouts(
                checkpoint=args.checkpoint,
                config_path=args.config,
                zarr_path=args.zarr,
                per_skill=args.per_skill,
                label_key=args.label_key,
                device=args.device,
                n_action_steps=args.n_action_steps,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
                record_video=not args.no_video,
                no_video=args.no_video,
            )
        else:
            run_representative_segment_rollouts_dp(
                dp_checkpoint=args.checkpoint,
                zarr_path=args.zarr,
                per_skill=args.per_skill,
                label_key=args.label_key,
                device=args.device,
                n_action_steps=args.n_action_steps,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
                record_video=not args.no_video,
            )
    else:
        if args.policy == "soda":
            run_segment_rollout_from_cli(
                checkpoint=args.checkpoint,
                config_path=args.config,
                zarr_path=args.zarr,
                segment_index=args.segment_index,
                label_key=args.label_key,
                device=args.device,
                n_action_steps=args.n_action_steps,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
                record_video=not args.no_video,
                no_video=args.no_video,
            )
        else:
            store = ZarrSegmentStore.open(args.zarr, label_key=args.label_key)
            segment = store.segment_by_index(args.segment_index)
            rollout_segment_dp(
                store,
                segment,
                segment_index=args.segment_index,
                dp_checkpoint=args.checkpoint,
                device=args.device,
                n_action_steps=args.n_action_steps,
                max_steps=args.max_steps,
                output_dir=args.output_dir,
                record_video=not args.no_video,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
