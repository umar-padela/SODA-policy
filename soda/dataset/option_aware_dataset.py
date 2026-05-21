"""
Option-aware PyTorch dataset for SODA (project_plan §4.2, §7 row 11).

Loads DP-format zarr, indexes contiguous option segments, and applies
``TemporalStretcher`` to action targets.

References
----------
- ``third_party/diffusion_policy/dataset/pusht_image_dataset.py`` — obs layout / normalizer keys
- ``project_plan.md`` §5 — zarr layout; β derived at load time
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike

from soda.dataset.temporal_stretch import TemporalStretcher

# Version string for optional processed caches (bump if derivation rule changes).
DERIVE_BETA_LABELS_VERSION = "v1_last_frame_of_segment"


@dataclass(frozen=True)
class OptionSegment:
    """One contiguous option run within a single demo episode."""

    start: int  # global frame index (inclusive)
    end: int  # global frame index (exclusive)
    option_id: int
    episode_idx: int

    @property
    def length(self) -> int:
        return self.end - self.start


def derive_beta_labels(
    option_ids: ArrayLike,
    episode_ends: ArrayLike,
) -> np.ndarray:
    """
    Derive termination supervision ``beta_label`` from discrete option ids.

    For each episode (as defined by cumulative ``episode_ends``), every
    contiguous run of the same option id is one skill segment. The **last
    frame of each segment** gets ``beta_label = 1``; all other frames get
    ``0``. This marks "skill completion / handoff" for BCE training of
    ``beta_omega(s)``.
    """
    options = np.asarray(option_ids).reshape(-1)
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)

    if options.ndim != 1:
        raise ValueError(f"option_ids must be 1-D, got shape {options.shape}")
    if ends.ndim != 1:
        raise ValueError(f"episode_ends must be 1-D, got shape {ends.shape}")
    if ends.size == 0:
        raise ValueError("episode_ends must be non-empty")
    if ends[-1] != options.shape[0]:
        raise ValueError(
            f"episode_ends[-1] ({ends[-1]}) must equal len(option_ids) "
            f"({options.shape[0]})"
        )
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be strictly increasing positive cumsums")

    beta = np.zeros(options.shape[0], dtype=np.float32)
    start = 0
    for ep_end in ends:
        segment = options[start:ep_end]
        n = segment.shape[0]
        if n > 0:
            change_at = np.flatnonzero(segment[:-1] != segment[1:])
            last_frames = np.concatenate([change_at, [n - 1]])
            beta[start + last_frames] = 1.0
        start = int(ep_end)

    return beta


def build_option_segment_index(
    option_ids: ArrayLike,
    episode_ends: ArrayLike,
    *,
    min_segment_len: int = 1,
    max_segment_len: int | None = None,
) -> list[OptionSegment]:
    """
    List every contiguous option-id run, never crossing episode boundaries.
    """
    options = np.asarray(option_ids).reshape(-1)
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if min_segment_len < 1:
        raise ValueError(f"min_segment_len must be >= 1, got {min_segment_len}")

    segments: list[OptionSegment] = []
    ep_start = 0
    for ep_idx, ep_end in enumerate(ends):
        ep_end = int(ep_end)
        ep_options = options[ep_start:ep_end]
        if ep_options.size == 0:
            ep_start = ep_end
            continue

        boundaries = np.flatnonzero(ep_options[:-1] != ep_options[1:]) + 1
        seg_starts = np.concatenate([[0], boundaries])
        seg_ends = np.concatenate([boundaries, [ep_options.size]])

        for rel_start, rel_end in zip(seg_starts, seg_ends):
            length = int(rel_end - rel_start)
            if length < min_segment_len:
                continue
            if max_segment_len is not None and length > max_segment_len:
                continue
            segments.append(
                OptionSegment(
                    start=ep_start + int(rel_start),
                    end=ep_start + int(rel_end),
                    option_id=int(ep_options[int(rel_start)]),
                    episode_idx=ep_idx,
                )
            )
        ep_start = ep_end

    return segments


def _episode_train_mask(
    n_episodes: int,
    *,
    val_ratio: float,
    seed: int,
    max_train_episodes: int | None,
) -> np.ndarray:
    """``True`` = train episode (mirrors DP ``get_val_mask`` + ``downsample_mask``)."""
    if n_episodes < 1:
        raise ValueError("n_episodes must be >= 1")

    rng = np.random.default_rng(seed)
    n_val = int(round(n_episodes * val_ratio))
    n_val = min(max(n_val, 0), n_episodes)
    val_mask = np.zeros(n_episodes, dtype=bool)
    if n_val > 0:
        val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
        val_mask[val_idxs] = True
    train_mask = ~val_mask

    if max_train_episodes is not None:
        train_idxs = np.flatnonzero(train_mask)
        if train_idxs.size > max_train_episodes:
            keep = rng.choice(
                train_idxs, size=max_train_episodes, replace=False
            )
            new_train = np.zeros(n_episodes, dtype=bool)
            new_train[keep] = True
            train_mask = new_train

    return train_mask


def _config_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class OptionLabeledZarrDataset:
    """
    PyTorch ``Dataset`` over a task zarr with one sample per option segment.

    Each item:
    - ``obs`` — DP-compatible dict (``image`` CHW float, ``agent_pos``), length ``n_obs_steps``
    - ``action`` — ``(horizon_stretch_max, D+1)`` after temporal stretch + duration channel
    - ``option_id``, ``beta_label``, ``segment_length`` metadata

    Observations are anchored at a random frame inside the segment (train split) so
    ``beta_label`` includes both 0 and 1 examples.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        option_id_key: str = "option_id_supervised",
        horizon_stretch_max: int | None = None,
        *,
        n_obs_steps: int = 2,
        min_segment_len: int = 1,
        val_ratio: float = 0.0,
        seed: int = 42,
        max_train_episodes: int | None = None,
        train: bool = True,
        random_anchor: bool = True,
    ) -> None:
        import zarr

        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.is_dir():
            raise FileNotFoundError(f"zarr not found: {self.zarr_path}")

        self.option_id_key = option_id_key
        self.n_obs_steps = int(n_obs_steps)
        self.min_segment_len = int(min_segment_len)
        self.random_anchor = bool(random_anchor)
        self._val_ratio = float(val_ratio)
        self._seed = int(seed)
        self._max_train_episodes = max_train_episodes
        self._rng = np.random.default_rng(seed)

        root = zarr.open(str(self.zarr_path), mode="r")
        data = root["data"]
        meta = root["meta"]

        if option_id_key not in data:
            raise KeyError(
                f"{option_id_key!r} missing in {self.zarr_path}/data; "
                f"available: {list(data.keys())}"
            )

        self._root = root
        self._img = data["img"]
        self._state = data["state"]
        self._action = data["action"]
        self._option_ids = np.asarray(data[option_id_key][:])
        self._episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
        self._beta_labels = derive_beta_labels(self._option_ids, self._episode_ends)

        all_segments = build_option_segment_index(
            self._option_ids,
            self._episode_ends,
            min_segment_len=self.min_segment_len,
        )
        if not all_segments:
            raise ValueError(f"No option segments found in {self.zarr_path}")

        max_seg_len = max(s.length for s in all_segments)
        if horizon_stretch_max is None:
            horizon_stretch_max = max_seg_len
        self.horizon_stretch_max = int(horizon_stretch_max)
        if max_seg_len > self.horizon_stretch_max:
            raise ValueError(
                f"longest segment ({max_seg_len}) exceeds horizon_stretch_max "
                f"({self.horizon_stretch_max}); increase horizon_stretch_max or "
                "filter segments"
            )

        self._stretcher = TemporalStretcher(horizon_stretch_max=self.horizon_stretch_max)

        n_episodes = int(self._episode_ends.size)
        train_ep_mask = _episode_train_mask(
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

    def get_validation_dataset(self) -> OptionLabeledZarrDataset:
        """Validation split (complement of train episode mask)."""
        return OptionLabeledZarrDataset(
            zarr_path=self.zarr_path,
            option_id_key=self.option_id_key,
            horizon_stretch_max=self.horizon_stretch_max,
            n_obs_steps=self.n_obs_steps,
            min_segment_len=self.min_segment_len,
            val_ratio=self._val_ratio,
            seed=self._seed,
            max_train_episodes=self._max_train_episodes,
            train=False,
            random_anchor=False,
        )

    def __len__(self) -> int:
        return len(self.segments)

    def _obs_window(self, seg: OptionSegment, anchor: int) -> dict[str, np.ndarray]:
        """``n_obs_steps`` frames ending at ``anchor`` (inclusive), padded at segment start."""
        win_end = anchor + 1
        win_start = max(seg.start, win_end - self.n_obs_steps)

        imgs: list[np.ndarray] = []
        states: list[np.ndarray] = []
        for idx in range(win_start, win_end):
            img = np.asarray(self._img[idx], dtype=np.uint8)
            imgs.append(np.moveaxis(img, -1, 0).astype(np.float32) / 255.0)
            states.append(np.asarray(self._state[idx, :2], dtype=np.float32))

        while len(imgs) < self.n_obs_steps:
            imgs.insert(0, imgs[0].copy())
            states.insert(0, states[0].copy())

        return {
            "image": np.stack(imgs[-self.n_obs_steps :], axis=0),
            "agent_pos": np.stack(states[-self.n_obs_steps :], axis=0),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        seg = self.segments[index]
        if self.random_anchor:
            anchor = int(self._rng.integers(seg.start, seg.end))
        else:
            anchor = seg.end - 1

        obs = self._obs_window(seg, anchor)
        actions_native = np.asarray(
            self._action[seg.start : seg.end], dtype=np.float32
        )
        action = self._stretcher.stretch_with_duration(actions_native)
        beta = float(self._beta_labels[anchor])

        return {
            "obs": {
                "image": torch.from_numpy(obs["image"]),
                "agent_pos": torch.from_numpy(obs["agent_pos"]),
            },
            "action": torch.from_numpy(action),
            "option_id": torch.tensor(seg.option_id, dtype=torch.long),
            "beta_label": torch.tensor(beta, dtype=torch.float32),
            "segment_length": torch.tensor(seg.length, dtype=torch.long),
            "segment_start": seg.start,
            "segment_end": seg.end,
            "anchor_index": anchor,
        }

    def get_option_segment_bounds(self) -> list[tuple[int, int, int]]:
        """``(start, end, option_id)`` per segment for ``train_high`` / flow matching."""
        return [(s.start, s.end, s.option_id) for s in self.segments]

    def get_normalizer(self, mode: str = "limits", **kwargs: Any) -> Any:
        """
        DP-compatible ``LinearNormalizer`` (requires ``diffusion_policy`` installed).
        """
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        from diffusion_policy.common.normalize_util import get_image_range_normalizer

        actions = np.asarray(self._action[:], dtype=np.float32)
        agent_pos = np.asarray(self._state[:, :2], dtype=np.float32)
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={"action": actions, "agent_pos": agent_pos},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        normalizer["image"] = get_image_range_normalizer()
        return normalizer


def build_option_dataset_from_config(cfg: Any) -> OptionLabeledZarrDataset:
    """
    Hydra-style factory: ``cfg.dataset`` (or flat ``cfg``) → ``OptionLabeledZarrDataset``.

    Expected keys: ``zarr_path``, ``option_id_key``, ``horizon_stretch_max``,
    ``n_obs_steps``, ``min_segment_len``, ``val_ratio``, ``seed``, ``max_train_episodes``.
    """
    ds_cfg = _config_get(cfg, "dataset", cfg)
    zarr_path = _config_get(ds_cfg, "zarr_path")
    if zarr_path is None:
        task = _config_get(ds_cfg, "task", "pusht")
        zarr_path = f"data/raw/{task}/{task}.zarr"

    dataset = OptionLabeledZarrDataset(
        zarr_path=zarr_path,
        option_id_key=_config_get(ds_cfg, "option_id_key", "option_id_supervised"),
        horizon_stretch_max=_config_get(ds_cfg, "horizon_stretch_max", None),
        n_obs_steps=int(_config_get(ds_cfg, "n_obs_steps", 2)),
        min_segment_len=int(_config_get(ds_cfg, "min_segment_len", 1)),
        val_ratio=float(_config_get(ds_cfg, "val_ratio", 0.0)),
        seed=int(_config_get(ds_cfg, "seed", 42)),
        max_train_episodes=_config_get(ds_cfg, "max_train_episodes", None),
        train=bool(_config_get(ds_cfg, "train", True)),
        random_anchor=bool(_config_get(ds_cfg, "random_anchor", True)),
    )
    return dataset
