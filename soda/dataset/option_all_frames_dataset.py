"""
All-frames dataset for π_high training (experimental_plan §10).

Unlike OptionStartDataset (one anchor per segment-start), this dataset includes
every frame from every segment with label = the option_id of that segment.

Trains π_high to answer "what skill is the expert executing in this state?" for
any frame — full state distribution coverage vs. the thin slice of segment-start
states in OptionStartDataset.
"""

from __future__ import annotations

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


class OptionAllFramesDataset:
    """
    One sample per (frame, option_id) for every frame in every segment.

    ``self.segments`` retains the list of unique OptionSegments (same as
    OptionStartDataset) so downstream code (num_options inference, class-weight
    computation) works without changes.

    ``self._samples`` is a list of ``(seg, anchor, null_token)`` 3-tuples —
    same structure as OptionStartDataset._samples so train_high.py helpers
    (e.g. _resolve_option_class_weights) work unchanged.

    prev_option_id is always the null token (= num_options) because
    condition_on_prev_option must be False for all-frames training: for an
    arbitrary mid-segment frame the previous option is not well-defined.
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

        null_token = self.num_options
        self._samples: list[tuple[OptionSegment, int, int]] = []
        for seg in self.segments:
            for frame_idx in range(seg.start, seg.end):
                self._samples.append((seg, frame_idx, null_token))

    def get_validation_dataset(self) -> OptionAllFramesDataset:
        return OptionAllFramesDataset(
            zarr_path=self.zarr_path,
            option_id_key=self.option_id_key,
            n_obs_steps=self.n_obs_steps,
            min_segment_len=self.min_segment_len,
            val_ratio=self._val_ratio,
            seed=self._seed,
            max_train_episodes=self._max_train_episodes,
            train=False,
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


def build_option_all_frames_dataset_from_config(cfg: Any) -> OptionAllFramesDataset:
    """Hydra-style factory: ``cfg.dataset`` (or flat ``cfg``) → ``OptionAllFramesDataset``."""
    ds_cfg = _config_get(cfg, "dataset", cfg)
    zarr_path = _config_get(ds_cfg, "zarr_path")
    if zarr_path is None:
        task = _config_get(ds_cfg, "task", "pusht")
        zarr_path = f"data/raw/{task}/{task}.zarr"

    return OptionAllFramesDataset(
        zarr_path=zarr_path,
        option_id_key=_config_get(ds_cfg, "option_id_key", "option_id_supervised"),
        n_obs_steps=int(_config_get(ds_cfg, "n_obs_steps", 2)),
        min_segment_len=int(_config_get(ds_cfg, "min_segment_len", 1)),
        val_ratio=float(_config_get(ds_cfg, "val_ratio", 0.0)),
        seed=int(_config_get(ds_cfg, "seed", 42)),
        max_train_episodes=_config_get(ds_cfg, "max_train_episodes", None),
        train=bool(_config_get(ds_cfg, "train", True)),
    )
