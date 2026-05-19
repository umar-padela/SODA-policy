#!/usr/bin/env python3
"""
Print label statistics and segment-length histograms for Push-T option labels.

Loads ``option_id_supervised`` from the task zarr (or recomputes from Columbia zip
with ``--relabel``). Same checks as the historical Colab notebook.

Example::

    python -m soda.option_discovery.supervised.pusht.visualize_labels
    python -m soda.option_discovery.supervised.pusht.visualize_labels --save figures/pusht_labels.png
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import zarr

from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES, label_skills

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"
DEFAULT_LABEL_KEY = "option_id_supervised"

SegmentRecord = tuple[int, int, int, int]  # ep_idx, length, frame_start, frame_end


def visualize_labels(
    labels: np.ndarray,
    episode_ends: np.ndarray,
    skill_names: Mapping[int, str] | None = None,
    *,
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """
    Aggregate stats, segment-length tables, short/long episode reports, histograms.
    """
    skill_names = skill_names or SKILL_NAMES
    labels = np.asarray(labels, dtype=np.int32)
    episode_ends = np.asarray(episode_ends, dtype=np.int32)

    total_points = len(labels)
    counts = Counter(labels.tolist())

    print("--- GLOBAL AGGREGATES ---")
    print(f"{'Skill':<15} | {'Points':<10} | {'Percentage':<10}")
    print("-" * 40)
    for s_id, name in skill_names.items():
        count = counts.get(s_id, 0)
        percent = (count / total_points) * 100 if total_points else 0.0
        print(f"{name:<15} | {count:<10} | {percent:>8.2f}%")
    print("-" * 40)
    print(f"{'TOTAL':<15} | {total_points:<10} | 100.00%\n")

    lengths: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    starts: dict[int, list[int]] = {0: [], 1: [], 2: [], 3: []}
    segment_map: dict[int, list[SegmentRecord]] = {1: [], 2: [], 3: []}

    start_idx = 0
    for ep_idx, ep_end in enumerate(episode_ends):
        ep_labels = labels[start_idx:ep_end]
        if len(ep_labels) == 0:
            start_idx = ep_end
            continue

        current_skill = int(ep_labels[0])
        current_len = 0
        current_segment_start = 0

        for i, label in enumerate(ep_labels):
            label = int(label)
            if label == current_skill:
                current_len += 1
            else:
                lengths[current_skill].append(current_len)
                starts[current_skill].append(current_segment_start)
                if current_skill in segment_map:
                    segment_map[current_skill].append(
                        (ep_idx, current_len, current_segment_start, i - 1)
                    )
                current_skill = label
                current_len = 1
                current_segment_start = i

        lengths[current_skill].append(current_len)
        starts[current_skill].append(current_segment_start)
        if current_skill in segment_map:
            segment_map[current_skill].append(
                (ep_idx, current_len, current_segment_start, len(ep_labels) - 1)
            )

        start_idx = ep_end

    print("--- SEGMENT LENGTH STATISTICS (Frames) ---")
    print(f"{'Skill':<15} | {'Mean':<6} | {'Std':<6} | {'Min':<5} | {'Max':<5}")
    print("-" * 50)
    for s_id, name in skill_names.items():
        data = lengths[s_id]
        if data:
            mu, std = float(np.mean(data)), float(np.std(data))
            mn, mx = int(np.min(data)), int(np.max(data))
            print(f"{name:<15} | {mu:>6.1f} | {std:>6.1f} | {mn:>5} | {mx:>5}")
        else:
            print(f"{name:<15} | N/A")
    print("-" * 50 + "\n")

    print("--- SHORT SEGMENT ANALYSIS ---")
    print(f"{'Skill':<15} | {'<2 fr %':<8} | {'<3 fr %':<8} | {'<4 fr %':<8} | {'Mid-Ep <4 %'}")
    print("-" * 65)
    for s_id, name in skill_names.items():
        l_data = np.array(lengths[s_id], dtype=np.int32)
        s_data = np.array(starts[s_id], dtype=np.int32)
        if len(l_data) > 0:
            p2 = (np.sum(l_data < 2) / len(l_data)) * 100
            p3 = (np.sum(l_data < 3) / len(l_data)) * 100
            mask_lt4 = l_data < 4
            p4 = (np.sum(mask_lt4) / len(l_data)) * 100
            short_not_at_0 = int(np.sum(mask_lt4 & (s_data != 0)))
            total_short = int(np.sum(mask_lt4))
            p4_mid = (short_not_at_0 / total_short * 100) if total_short > 0 else 0.0
            print(f"{name:<15} | {p2:>7.1f}% | {p3:>7.1f}% | {p4:>7.1f}% | {p4_mid:>10.1f}%")
        else:
            print(f"{name:<15} | N/A")
    print("-" * 65 + "\n")

    def _format_episode_report(
        segments: list[SegmentRecord], *, longest: bool
    ) -> list[str]:
        ordered = sorted(segments, key=lambda x: x[1], reverse=longest)
        unique_report: list[str] = []
        seen_eps: set[int] = set()
        for ep_i, length, f_start, f_end in ordered:
            if ep_i in seen_eps:
                continue
            unique_report.append(f"Ep {ep_i} frame {f_start}-{f_end} (len {length})")
            seen_eps.add(ep_i)
            if len(unique_report) >= 3:
                break
        return unique_report

    print("--- EPISODES WITH SHORTEST SEGMENTS ---")
    for s_id in (1, 2, 3):
        name = skill_names[s_id]
        report = _format_episode_report(segment_map[s_id], longest=False)
        print(f"{name:<12}: " + (", ".join(report) if report else "(none)"))
    print("")

    print("--- EPISODES WITH LONGEST SEGMENTS ---")
    for s_id in (1, 2, 3):
        name = skill_names[s_id]
        report = _format_episode_report(segment_map[s_id], longest=True)
        print(f"{name:<12}: " + (", ".join(report) if report else "(none)"))
    print("")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()
    colors = ["gray", "blue", "green", "orange"]

    for i, (s_id, name) in enumerate(skill_names.items()):
        data = lengths[s_id]
        ax = axes_flat[i]
        if data:
            ax.hist(data, bins=30, color=colors[i], edgecolor="black", alpha=0.7)
            ax.set_title(f"Length Distribution: {name}")
            ax.set_xlabel("Frames")
            ax.set_ylabel("Frequency")
        else:
            ax.set_title(f"{name} (No Data)")

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    repo_segments = segment_map.get(1, [])
    target = next((seg for seg in repo_segments if seg[1] == 3), None)
    if target is not None:
        print(f"Example 3-frame REPOSITION segment: Ep {target[0]} frames {target[2]}-{target[3]}")
    else:
        print("No 3-frame REPOSITION segment found.")


def load_from_zarr(zarr_path: Path, label_key: str) -> tuple[np.ndarray, np.ndarray]:
    root = zarr.open(str(zarr_path), mode="r")
    labels = np.asarray(root["data"][label_key][:], dtype=np.int32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int32)
    return labels, episode_ends


def load_relabeled_from_columbia(zip_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from soda.option_discovery.supervised.pusht.build_zarr import load_flat_dataset

    dataset = load_flat_dataset(zip_path)
    labels = label_skills(dataset)
    return labels, dataset["episode_ends"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=DEFAULT_ZARR,
        help="Push-T zarr directory store",
    )
    parser.add_argument(
        "--label-key",
        default=DEFAULT_LABEL_KEY,
        help="Option label array under data/ (default: option_id_supervised)",
    )
    parser.add_argument(
        "--relabel",
        action="store_true",
        help="Ignore zarr labels; recompute from Columbia zip via build_zarr.load_flat_dataset",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "pusht_cache" / "pusht_cchi_v7_replay.zarr.zip",
        help="Columbia zip (only with --relabel)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save histogram figure to this path",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive plot window (use with --save on headless machines)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.relabel:
        if not args.zip_path.is_file():
            print(f"Missing Columbia zip: {args.zip_path}", file=sys.stderr)
            return 1
        labels, episode_ends = load_relabeled_from_columbia(args.zip_path)
        print(f"Relabeled from zip: {args.zip_path}\n")
    else:
        if not args.zarr_path.is_dir():
            print(f"Missing zarr: {args.zarr_path}", file=sys.stderr)
            return 1
        labels, episode_ends = load_from_zarr(args.zarr_path, args.label_key)
        print(f"Loaded {args.zarr_path} [{args.label_key}]\n")

    visualize_labels(
        labels,
        episode_ends,
        show=not args.no_show,
        save_path=args.save,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
