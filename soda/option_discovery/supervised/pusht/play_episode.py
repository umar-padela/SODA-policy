#!/usr/bin/env python3
"""
Export (and optionally play) a Push-T episode video with skill / state overlays.

Example::

    python -m soda.option_discovery.supervised.pusht.play_episode --episode 100
    python -m soda.option_discovery.supervised.pusht.play_episode --episode 0 --output /tmp/ep0.mp4 --open
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from soda.option_discovery.supervised.pusht.episode_video import render_episode_video

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"
DEFAULT_LABEL_KEY = "option_id_supervised"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=100, help="Episode index")
    parser.add_argument("--zarr-path", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--label-key", default=DEFAULT_LABEL_KEY)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 (default: tmp/pusht_videos/episode_{idx}.mp4, gitignored)",
    )
    parser.add_argument(
        "--open",
        dest="open_player",
        action="store_true",
        help="Open video with system default player after export",
    )
    parser.add_argument(
        "--no-burn-state",
        action="store_true",
        help="Skip state/skill overlay (frame numbers only if --burn-frame-numbers)",
    )
    parser.add_argument(
        "--no-burn-frame-numbers",
        action="store_true",
        help="Skip frame index overlay",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.zarr_path.is_dir():
        print(f"Missing zarr: {args.zarr_path}", file=sys.stderr)
        return 1

    out = args.output
    if out is None:
        out = REPO_ROOT / "tmp" / "pusht_videos" / f"episode_{args.episode}.mp4"

    render_episode_video(
        args.zarr_path,
        args.episode,
        out,
        label_key=args.label_key,
        burn_frame_numbers=not args.no_burn_frame_numbers,
        burn_state=not args.no_burn_state,
        open_player=args.open_player,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
