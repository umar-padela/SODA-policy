"""
Option segment-start PyTorch dataset for π_high training (project_plan §7 row 11).

Default: one sample per segment (anchor = segment.start).
With ``cut_last_n > 0`` (train only): enumerate every frame in
``[seg.start, seg.end - cut_last_n)`` as a valid anchor, giving
``max(1, seg_len - cut_last_n)`` samples per segment.  Val always
uses a single anchor at segment.start for a stable, reproducible metric.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from soda.dataset.option_aware_dataset import (
    OptionSegment,
    _config_get,
    build_option_segment_index,
    episode_train_mask,
    obs_window_at_anchor,
)


class OptionStartDataset:
    """
    Samples for π_high: obs window at anchor frame within a segment, label ω.

    Training (``cut_last_n > 0``): enumerates multiple anchors per segment
    (``max(1, seg_len - cut_last_n)`` anchors), skipping the last N frames
    which are near option boundaries and may be visually ambiguous.
    Validation: always a single anchor at segment.start regardless of cut_last_n.

    ``self.segments`` retains the list of unique OptionSegments for downstream
    use (option ID counting, class weight computation).
    """

    def __init__(
        self,
        zarr_path: str | Path,
        option_id_key: str = "option_id_supervised",
        *,
        n_obs_steps: int = 2,
        min_segment_len: int = 1,
        val_ratio: float = 0.0,
        seed: int = 42,
        max_train_episodes: int | None = None,
        train: bool = True,
        cut_last_n: int = 0,
    ) -> None:
        import zarr

        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.is_dir():
            raise FileNotFoundError(f"zarr not found: {self.zarr_path}")

        self.option_id_key = option_id_key
        self.n_obs_steps = int(n_obs_steps)
        self.min_segment_len = int(min_segment_len)
        self._val_ratio = float(val_ratio)
        self._seed = int(seed)
        self._max_train_episodes = max_train_episodes
        self._cut_last_n = int(cut_last_n)
        self._train = train

        root = zarr.open(str(self.zarr_path), mode="r")
        data = root["data"]
        meta = root["meta"]

        if option_id_key not in data:
            raise KeyError(
                f"{option_id_key!r} missing in {self.zarr_path}/data; "
                f"available: {list(data.keys())}"
            )

        self._img = data["img"]
        self._state = data["state"]
        self._option_ids = np.asarray(data[option_id_key][:])
        self._episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)

        all_segments = build_option_segment_index(
            self._option_ids,
            self._episode_ends,
            min_segment_len=self.min_segment_len,
        )
        if not all_segments:
            raise ValueError(f"No option segments found in {self.zarr_path}")

        # num_options inferred from all segments (before train/val split) so both
        # splits agree on the null-token index (= num_options).
        all_option_ids = {s.option_id for s in all_segments}
        self.num_options: int = len(all_option_ids)

        n_episodes = int(self._episode_ends.size)
        train_ep_mask = episode_train_mask(
            n_episodes,
            val_ratio=val_ratio,
            seed=seed,
            max_train_episodes=max_train_episodes,
        )
        use_train_eps = train_ep_mask if train else ~train_ep_mask
        self.segments = [s for s in all_segments if use_train_eps[s.episode_idx]]
        if not self.segments:
            raise ValueError(
                f"No segments after {'train' if train else 'val'} episode mask"
            )

        # Build prev_option_id per segment.
        # Within each episode, segments are sorted by start position; the first
        # segment uses the null token (index = num_options = "no previous option").
        null_token = self.num_options
        ep_segs: dict[int, list[OptionSegment]] = defaultdict(list)
        for seg in self.segments:
            ep_segs[seg.episode_idx].append(seg)
        prev_option_by_key: dict[tuple[int, int], int] = {}
        for ep_idx, segs in ep_segs.items():
            segs_sorted = sorted(segs, key=lambda s: s.start)
            for i, seg in enumerate(segs_sorted):
                prev_option_by_key[(seg.episode_idx, seg.start)] = (
                    null_token if i == 0 else segs_sorted[i - 1].option_id
                )

        # Build (segment, anchor, prev_option_id) sample index.
        # Val always uses single anchor at segment.start (stable, reproducible metric).
        # Train with cut_last_n > 0 enumerates max(1, seg_len - cut_last_n) anchors.
        effective_cut = self._cut_last_n if train else 0
        self._samples: list[tuple[OptionSegment, int, int]] = []
        for seg in self.segments:
            prev_opt = prev_option_by_key[(seg.episode_idx, seg.start)]
            seg_len = seg.end - seg.start
            n_anchors = max(1, seg_len - effective_cut) if effective_cut > 0 else 1
            for offset in range(n_anchors):
                self._samples.append((seg, seg.start + offset, prev_opt))

    def get_validation_dataset(self) -> OptionStartDataset:
        """Validation split (complement of train episode mask). Single anchor per segment."""
        return OptionStartDataset(
            zarr_path=self.zarr_path,
            option_id_key=self.option_id_key,
            n_obs_steps=self.n_obs_steps,
            min_segment_len=self.min_segment_len,
            val_ratio=self._val_ratio,
            seed=self._seed,
            max_train_episodes=self._max_train_episodes,
            train=False,
            cut_last_n=self._cut_last_n,  # stored for round-trip; val ignores it
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seg, anchor, prev_opt = self._samples[index]
        obs = obs_window_at_anchor(
            self._img, self._state, seg, anchor, self.n_obs_steps
        )

        return {
            "obs": {
                "image": torch.from_numpy(obs["image"]),
                "agent_pos": torch.from_numpy(obs["agent_pos"]),
            },
            "option_id": torch.tensor(seg.option_id, dtype=torch.long),
            "prev_option_id": torch.tensor(prev_opt, dtype=torch.long),
            "segment_start": seg.start,
            "segment_end": seg.end,
            "anchor_index": anchor,
        }


def build_option_start_dataset_from_config(cfg: Any) -> OptionStartDataset:
    """Hydra-style factory: ``cfg.dataset`` (or flat ``cfg``) → ``OptionStartDataset``."""
    ds_cfg = _config_get(cfg, "dataset", cfg)
    zarr_path = _config_get(ds_cfg, "zarr_path")
    if zarr_path is None:
        task = _config_get(ds_cfg, "task", "pusht")
        zarr_path = f"data/raw/{task}/{task}.zarr"

    return OptionStartDataset(
        zarr_path=zarr_path,
        option_id_key=_config_get(ds_cfg, "option_id_key", "option_id_supervised"),
        n_obs_steps=int(_config_get(ds_cfg, "n_obs_steps", 2)),
        min_segment_len=int(_config_get(ds_cfg, "min_segment_len", 1)),
        val_ratio=float(_config_get(ds_cfg, "val_ratio", 0.0)),
        seed=int(_config_get(ds_cfg, "seed", 42)),
        max_train_episodes=_config_get(ds_cfg, "max_train_episodes", None),
        train=bool(_config_get(ds_cfg, "train", True)),
        cut_last_n=int(_config_get(ds_cfg, "anchor_cut_last_n", 0)),
    )
