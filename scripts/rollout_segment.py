#!/usr/bin/env python3
"""
Expert-anchored π_low rollout: reset sim to a zarr segment start, roll out with fixed ω.

Examples (repo root, GPU + diffusion_policy submodule)::

    # List REPOSITION segments in episode 100
    python scripts/rollout_segment.py --list-segments --episode 100 --option-id 0

    # Rollout first REPOSITION segment in episode 100; side-by-side expert | policy MP4
    python scripts/rollout_segment.py \\
        --config configs/pusht/soda_supervised.yaml \\
        --checkpoint experiments/train_low/soda_supervised/best.ckpt \\
        --episode 100 --option-id 0

    # By global segment index (from --list-segments)
    python scripts/rollout_segment.py \\
        --config configs/pusht/soda_supervised.yaml \\
        --checkpoint experiments/train_low/soda_supervised/best.ckpt \\
        --segment-index 42

Modal (Volume checkpoint + GPU; no local download)::

    modal run modal/modal_rollout_low_policy.py --run-readme "ep100 o0 qual" \\
        --episode 100 --option-id 0

Skill ids: 0 REPOSITION, 1 LINEAR-PUSH, 2 PIVOT-PUSH.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from soda.eval.eval_yaml import resolve_eval_config_path
from soda.experiments.paths import infer_task_slug, segment_rollout_dir
from soda.eval.segment_rollout import (
    run_representative_segment_rollouts,
    run_segment_rollout_from_cli,
)
from soda.eval.segment_zarr import (
    DEFAULT_LABEL_KEY,
    DEFAULT_ZARR,
    REPRESENTATIVE_MIN_SEGMENT_LENGTH,
    list_segments_text,
    ZarrSegmentStore,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pusht/soda_supervised.yaml"),
        help="Eval/train yaml (obs steps, env_runner max_steps cap)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="π_low checkpoint (best.ckpt or epoch_*.ckpt)",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=DEFAULT_ZARR,
        help="Task zarr with option labels",
    )
    parser.add_argument(
        "--label-key",
        default=DEFAULT_LABEL_KEY,
        help="Option label array in zarr (option_id_supervised | option_id_unsupervised)",
    )
    parser.add_argument(
        "--list-segments",
        action="store_true",
        help="Print matching segments and exit",
    )
    parser.add_argument(
        "--representative-samples",
        action="store_true",
        help="Auto-pick per_skill median-length segments for each skill (default 2 → 6 rollouts)",
    )
    parser.add_argument(
        "--per-skill",
        type=int,
        default=2,
        help="Segments per skill when --representative-samples (default 2)",
    )
    parser.add_argument(
        "--segment-index",
        type=int,
        default=None,
        help="Global segment index (see --list-segments)",
    )
    parser.add_argument("--episode", type=int, default=None, help="Episode index in zarr")
    parser.add_argument(
        "--option-id",
        type=int,
        default=None,
        help="Skill id 0/1/2 (required with --episode unless --segment-index set)",
    )
    parser.add_argument(
        "--segment-rank",
        type=int,
        default=0,
        help="Nth segment matching --episode + --option-id (default 0)",
    )
    parser.add_argument(
        "--fixed-option-id",
        type=int,
        default=None,
        help="Override ω passed to π_low (default: segment option_id)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Receding-horizon execute window (match eval yaml)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap rollout length (default: max(segment_length, 50) — policy always gets at least 50 frames)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for MP4 + metrics JSON (default: experiments/{task}/segment_rollout/)",
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--list-limit",
        type=int,
        default=30,
        help="Max rows when --list-segments",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = resolve_eval_config_path(args.config)
    output_dir = args.output_dir or Path(
        segment_rollout_dir(infer_task_slug(config_path=config_path))
    )

    if args.list_segments:
        store = ZarrSegmentStore.open(args.zarr_path, label_key=args.label_key)
        print(
            list_segments_text(
                store,
                option_id=args.option_id,
                episode_idx=args.episode,
                limit=args.list_limit,
            )
        )
        return 0

    if args.checkpoint is None:
        print("error: --checkpoint is required for rollout", file=sys.stderr)
        return 2
    if not args.representative_samples and args.segment_index is None and (
        args.episode is None or args.option_id is None
    ):
        print(
            "error: pass --representative-samples, --segment-index, "
            "OR (--episode and --option-id)",
            file=sys.stderr,
        )
        return 2

    if args.representative_samples:
        results = run_representative_segment_rollouts(
            checkpoint=args.checkpoint,
            config_path=config_path,
            zarr_path=args.zarr_path,
            per_skill=args.per_skill,
            min_segment_length=REPRESENTATIVE_MIN_SEGMENT_LENGTH,
            fixed_option_id=args.fixed_option_id,
            label_key=args.label_key,
            device=args.device,
            n_action_steps=args.n_action_steps,
            max_steps=args.max_steps,
            output_dir=output_dir,
            no_video=args.no_video,
        )
        print(f"Representative segment rollouts complete ({len(results)} total)")
        for result in results:
            m = result.metrics
            print(
                f"  segment_index={result.segment_index} ep={m['episode_idx']} "
                f"ω={m['option_id']} ({m['skill']}) steps={result.n_policy_steps} "
                f"max_overlap={m.get('max_overlap_full', 0):.1f}%"
            )
            if result.video_path is not None:
                print(f"    video={result.video_path}")
        return 0

    result = run_segment_rollout_from_cli(
        checkpoint=args.checkpoint,
        config_path=config_path,
        zarr_path=args.zarr_path,
        segment_index=args.segment_index,
        episode_idx=args.episode,
        option_id=args.option_id,
        segment_rank=args.segment_rank,
        fixed_option_id=args.fixed_option_id,
        label_key=args.label_key,
        device=args.device,
        n_action_steps=args.n_action_steps,
        max_steps=args.max_steps,
        output_dir=output_dir,
        no_video=args.no_video,
    )

    m = result.metrics
    print("Segment rollout complete")
    print(
        f"  segment_index={result.segment_index} ep={m['episode_idx']} "
        f"ω={m['option_id']} ({m['skill']})"
    )
    print(f"  anchor_frame={result.anchor_frame} steps={result.n_policy_steps}")
    print(
        f"  max_overlap_full={m.get('max_overlap_full', 0):.1f}% "
        f"success={m.get('success', False)}"
    )
    if result.video_path is not None:
        print(f"  video={result.video_path}")
        json_path = result.video_path.with_suffix(".json")
        if json_path.is_file():
            print(f"  metrics={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
