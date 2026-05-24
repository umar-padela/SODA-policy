"""
Load expert option segments from Push-T zarr for anchored π_low rollouts.

Segments match ``OptionAwareDataset`` indexing (contiguous ``option_id`` runs
within episodes). The anchor frame is ``segment.start`` — same as training
when ``random_anchor=False`` on val.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from soda.dataset.option_aware_dataset import OptionSegment, build_option_segment_index
from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"
DEFAULT_LABEL_KEY = "option_id_supervised"
# Representative rollouts: segment length must exceed this (frames).
REPRESENTATIVE_MIN_SEGMENT_LENGTH = 26


@dataclass(frozen=True)
class ZarrSegmentStore:
    """Read-only view of zarr arrays needed for segment rollouts."""

    zarr_path: Path
    label_key: str
    option_ids: np.ndarray
    episode_ends: np.ndarray
    state: zarr.Array
    img: zarr.Array
    segments: list[OptionSegment]

    @classmethod
    def open(
        cls,
        zarr_path: str | Path,
        *,
        label_key: str = DEFAULT_LABEL_KEY,
        min_segment_len: int = 1,
    ) -> ZarrSegmentStore:
        zarr_path = Path(zarr_path)
        if not zarr_path.is_dir():
            raise FileNotFoundError(f"zarr not found: {zarr_path}")

        root = zarr.open(str(zarr_path), mode="r")
        if label_key not in root["data"]:
            raise KeyError(
                f"{label_key!r} missing in {zarr_path}/data; "
                "run option discovery first."
            )

        option_ids = np.asarray(root["data"][label_key][:], dtype=np.int32)
        episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        segments = build_option_segment_index(
            option_ids,
            episode_ends,
            min_segment_len=min_segment_len,
        )
        return cls(
            zarr_path=zarr_path,
            label_key=label_key,
            option_ids=option_ids,
            episode_ends=episode_ends,
            state=root["data"]["state"],
            img=root["data"]["img"],
            segments=segments,
        )

    def filter_segments(
        self,
        *,
        option_id: int | None = None,
        episode_idx: int | None = None,
        min_length: int = 1,
    ) -> list[OptionSegment]:
        out: list[OptionSegment] = []
        for seg in self.segments:
            if option_id is not None and seg.option_id != int(option_id):
                continue
            if episode_idx is not None and seg.episode_idx != int(episode_idx):
                continue
            if seg.length < int(min_length):
                continue
            out.append(seg)
        return out

    def segment_by_index(self, segment_index: int) -> OptionSegment:
        idx = int(segment_index)
        if idx < 0 or idx >= len(self.segments):
            raise IndexError(
                f"segment_index {idx} out of range [0, {len(self.segments) - 1}]"
            )
        return self.segments[idx]

    def pusht_env_state(self, global_frame: int) -> np.ndarray:
        """
        Push-T sim state vector for ``PushTEnv._set_state``.

        Zarr ``state`` is ``[agent_x, agent_y, block_x, block_y, block_angle_rad]``
        where block_angle_rad = block.angle % (2π) recorded by _get_obs() at collection
        time. Pass it unchanged to _set_state so the sim reproduces the exact recorded
        configuration.
        """
        vec = np.asarray(self.state[int(global_frame)], dtype=np.float64).reshape(-1)
        if vec.shape[0] != 5:
            raise ValueError(f"expected state dim 5, got {vec.shape}")
        return vec.copy()

    def expert_rgb_frame(self, global_frame: int) -> np.ndarray:
        """Single RGB frame ``(H, W, 3)`` uint8 from zarr."""
        return np.asarray(self.img[int(global_frame)], dtype=np.uint8)

    def expert_rgb_window(
        self,
        start_frame: int,
        n_frames: int,
    ) -> list[np.ndarray]:
        """Consecutive expert frames from ``start_frame`` (inclusive)."""
        if n_frames <= 0:
            return []
        end = min(int(start_frame) + int(n_frames), int(self.option_ids.shape[0]))
        return [self.expert_rgb_frame(i) for i in range(int(start_frame), end)]


def format_segment_line(seg: OptionSegment, *, index: int) -> str:
    skill = SKILL_NAMES.get(seg.option_id, str(seg.option_id))
    return (
        f"[{index:4d}] ep={seg.episode_idx:3d} ω={seg.option_id} ({skill:12s}) "
        f"frames [{seg.start:6d}, {seg.end:6d}) len={seg.length:3d}"
    )


def list_segments_text(
    store: ZarrSegmentStore,
    *,
    option_id: int | None = None,
    episode_idx: int | None = None,
    limit: int | None = 50,
) -> str:
    """Human-readable segment listing for CLI ``--list-segments``."""
    lines = [
        f"zarr={store.zarr_path} label={store.label_key} "
        f"total_segments={len(store.segments)}",
        "index | episode | ω | skill        | global frame range      | len",
        "-" * 72,
    ]
    shown = 0
    for i, seg in enumerate(store.segments):
        if option_id is not None and seg.option_id != int(option_id):
            continue
        if episode_idx is not None and seg.episode_idx != int(episode_idx):
            continue
        lines.append(format_segment_line(seg, index=i))
        shown += 1
        if limit is not None and shown >= int(limit):
            extra = sum(
                1
                for s in store.segments[i + 1 :]
                if (option_id is None or s.option_id == int(option_id))
                and (episode_idx is None or s.episode_idx == int(episode_idx))
            )
            if extra > 0:
                lines.append(f"... and {extra} more matching segments")
            break
    if shown == 0:
        lines.append("(no segments match filters)")
    return "\n".join(lines)


def select_representative_segments(
    store: ZarrSegmentStore,
    *,
    per_skill: int = 2,
    min_length: int = 8,
    max_length: int | None = None,
    skill_ids: list[int] | None = None,
) -> list[tuple[int, OptionSegment]]:
    """
    Pick ``per_skill`` segments per skill with length near the skill median.

    Prefers segments from different episodes when possible. Returns
    ``(global_segment_index, segment)`` pairs sorted by skill id then rank.
    """
    skill_ids = list(SKILL_NAMES.keys()) if skill_ids is None else list(skill_ids)
    picked: list[tuple[int, OptionSegment]] = []

    for oid in skill_ids:
        candidates: list[tuple[int, OptionSegment]] = []
        for idx, seg in enumerate(store.segments):
            if seg.option_id != int(oid):
                continue
            if seg.length < int(min_length):
                continue
            if max_length is not None and seg.length > int(max_length):
                continue
            candidates.append((idx, seg))

        if len(candidates) < int(per_skill):
            skill = SKILL_NAMES.get(int(oid), str(oid))
            raise ValueError(
                f"Need {per_skill} segments for skill {oid} ({skill}), "
                f"found {len(candidates)} with length>={min_length}"
            )

        median_len = float(np.median([seg.length for _, seg in candidates]))
        ranked = sorted(
            candidates,
            key=lambda item: (
                abs(item[1].length - median_len),
                item[1].episode_idx,
                item[0],
            ),
        )

        skill_picks: list[tuple[int, OptionSegment]] = []
        used_episodes: set[int] = set()
        for idx, seg in ranked:
            if len(skill_picks) >= int(per_skill):
                break
            if seg.episode_idx in used_episodes and len(skill_picks) < int(per_skill) - 1:
                continue
            skill_picks.append((idx, seg))
            used_episodes.add(seg.episode_idx)

        if len(skill_picks) < int(per_skill):
            for idx, seg in ranked:
                if all(idx != existing_idx for existing_idx, _ in skill_picks):
                    skill_picks.append((idx, seg))
                if len(skill_picks) >= int(per_skill):
                    break

        picked.extend(skill_picks[: int(per_skill)])

    return picked


def resolve_segment(
    store: ZarrSegmentStore,
    *,
    segment_index: int | None = None,
    episode_idx: int | None = None,
    option_id: int | None = None,
    segment_rank: int = 0,
) -> tuple[OptionSegment, int]:
    """
    Pick a segment and return ``(segment, global_segment_index)``.

    Either pass ``segment_index`` directly, or ``(episode_idx, option_id, segment_rank)``
    to select the n-th matching segment within an episode.
    """
    if segment_index is not None:
        seg = store.segment_by_index(segment_index)
        return seg, int(segment_index)

    if episode_idx is None or option_id is None:
        raise ValueError(
            "Pass --segment-index, or both --episode and --option-id "
            "(with optional --segment-rank)."
        )

    matches = store.filter_segments(
        option_id=int(option_id),
        episode_idx=int(episode_idx),
    )
    if not matches:
        raise ValueError(
            f"No segments for episode={episode_idx} option_id={option_id} "
            f"in {store.zarr_path}"
        )
    rank = int(segment_rank)
    if rank < 0 or rank >= len(matches):
        raise IndexError(
            f"segment_rank {rank} out of range [0, {len(matches) - 1}] "
            f"for episode={episode_idx} option_id={option_id}"
        )
    seg = matches[rank]
    global_idx = store.segments.index(seg)
    return seg, global_idx
