"""
Option-aware PyTorch dataset for π_low training (project_plan §4.2).

Loads DP-format zarr, indexes contiguous option segments, and applies
``TemporalStretcher`` to action targets. Shared segment/obs helpers are
also used by ``option_start_dataset`` (π_high).

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
DERIVE_BETA_LABELS_VERSION = "v2_binary_last_frame"


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
    Binary termination label: 0.0 for every frame except the last frame of each
    option segment, which gets 1.0.

    Binary labels force β to learn visual terminal-state recognition (not temporal
    fraction), which is the correct inductive bias for obs-based termination.
    Label smoothing in the loss prevents logit saturation; pos_weight handles the
    class imbalance (~1 positive per segment of average length 17).
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
    ep_start = 0
    for ep_end in ends:
        segment = options[ep_start:ep_end]
        n = segment.shape[0]
        if n > 0:
            change_at = np.flatnonzero(segment[:-1] != segment[1:])
            seg_ends_rel = np.concatenate([change_at, [n - 1]])
            for seg_end_rel in seg_ends_rel:
                beta[ep_start + int(seg_end_rel)] = 1.0
        ep_start = int(ep_end)

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


def native_actions_from_anchor(
    action: ArrayLike,
    seg: OptionSegment,
    anchor: int,
) -> np.ndarray:
    """
    Expert actions from ``anchor`` (inclusive) through ``seg.end`` (exclusive).

    Used for π_low training: obs at ``anchor`` supervises the **remaining** skill
    trajectory (suffix), not actions before ``anchor``.
    """
    if anchor < seg.start or anchor >= seg.end:
        raise ValueError(
            f"anchor {anchor} must lie in [{seg.start}, {seg.end}) for segment"
        )
    return np.asarray(action[anchor : seg.end], dtype=np.float32)


def obs_window_at_anchor(
    img: Any,
    state: Any,
    seg: OptionSegment,
    anchor: int,
    n_obs_steps: int,
) -> dict[str, np.ndarray]:
    """``n_obs_steps`` frames ending at ``anchor`` (inclusive), padded at segment start."""
    win_end = anchor + 1
    win_start = max(seg.start, win_end - n_obs_steps)

    imgs: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for idx in range(win_start, win_end):
        frame = np.asarray(img[idx], dtype=np.uint8)
        imgs.append(np.moveaxis(frame, -1, 0).astype(np.float32) / 255.0)
        states.append(np.asarray(state[idx, :2], dtype=np.float32))

    while len(imgs) < n_obs_steps:
        imgs.insert(0, imgs[0].copy())
        states.insert(0, states[0].copy())

    return {
        "image": np.stack(imgs[-n_obs_steps:], axis=0),
        "agent_pos": np.stack(states[-n_obs_steps:], axis=0),
    }


def episode_train_mask(
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


class OptionAwareDataset:
    """
    PyTorch ``Dataset`` over a task zarr with one sample per option segment.

    Each item:
    - ``obs`` — DP-compatible dict (``image`` CHW float, ``agent_pos``), length ``n_obs_steps``
    - ``action`` — ``(horizon, D+1)`` after temporal stretch + duration channel
    - ``option_id``, ``beta_label``, ``segment_length`` (remaining native steps at anchor)

    Observations are anchored at a random frame inside the segment (train split).
    Action targets are the **suffix** ``action[anchor:seg.end]`` (stretched to
    ``horizon``), aligned with closed-loop replanning from arbitrary mid-skill states.
    ``beta_label`` is ``1`` only when ``anchor`` is the segment's last frame.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        option_id_key: str = "option_id_supervised",
        horizon: int | None = None,
        *,
        n_obs_steps: int = 2,
        min_segment_len: int = 1,
        val_ratio: float = 0.0,
        seed: int = 42,
        max_train_episodes: int | None = None,
        train: bool = True,
        random_anchor: bool = True,
        all_anchors: bool = False,
        escape_relabeling: bool = False,
    ) -> None:
        import zarr

        self.zarr_path = Path(zarr_path)
        if not self.zarr_path.is_dir():
            raise FileNotFoundError(f"zarr not found: {self.zarr_path}")

        self.option_id_key = option_id_key
        self.n_obs_steps = int(n_obs_steps)
        self.min_segment_len = int(min_segment_len)
        self.random_anchor = bool(random_anchor)
        self._all_anchors = bool(all_anchors)
        self._stratified_encoded_indices = False
        self._val_ratio = float(val_ratio)
        self._seed = int(seed)
        self._max_train_episodes = max_train_episodes
        self._rng = np.random.default_rng(seed)
        # escape_relabeling only applies to train split — silently disabled for val.
        self.escape_relabeling = bool(escape_relabeling) and bool(train)

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
        self.num_options = int(len(np.unique(self._option_ids)))

        all_segments = build_option_segment_index(
            self._option_ids,
            self._episode_ends,
            min_segment_len=self.min_segment_len,
        )
        if not all_segments:
            raise ValueError(f"No option segments found in {self.zarr_path}")

        max_seg_len = max(s.length for s in all_segments)
        if horizon is None:
            horizon = max_seg_len
        self.horizon = int(horizon)
        if max_seg_len > self.horizon:
            raise ValueError(
                f"longest segment ({max_seg_len}) exceeds horizon "
                f"({self.horizon}); increase horizon or filter segments"
            )

        self._stretcher = TemporalStretcher(horizon=self.horizon)

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

        if self.escape_relabeling and not self._all_anchors:
            raise ValueError(
                "escape_relabeling=True requires all_anchors=True so every "
                "(segment, anchor) pair is enumerated and expanded."
            )

        # all_anchors: enumerate every valid (segment, anchor_frame) pair up front.
        # With escape_relabeling, each real pair is expanded to K tuples:
        #   (seg_idx, anchor, option_id_override, is_escape)
        # Without escape_relabeling, 2-tuples are used for backward compatibility.
        if self._all_anchors:
            if self.escape_relabeling:
                self._samples: list[tuple] = []
                for seg_idx, seg in enumerate(self.segments):
                    for anchor in range(seg.start, seg.end):
                        # Real sample (correct option)
                        self._samples.append((seg_idx, anchor, seg.option_id, False))
                        # K-1 escape samples (wrong option → β=1)
                        for wrong_opt in range(self.num_options):
                            if wrong_opt != seg.option_id:
                                self._samples.append((seg_idx, anchor, wrong_opt, True))
            else:
                self._samples = [
                    (seg_idx, anchor)
                    for seg_idx, seg in enumerate(self.segments)
                    for anchor in range(seg.start, seg.end)
                ]
        else:
            self._samples = []

    def enable_stratified_indices(self, enabled: bool = True) -> None:
        """
        When True, ``__getitem__`` accepts encoded indices from
        :mod:`option_stratified_sampler` (segment-end vs random anchor).
        """
        self._stratified_encoded_indices = bool(enabled)

    def get_validation_dataset(self, *, all_anchors: bool = False) -> OptionAwareDataset:
        """Validation split (complement of train episode mask).

        Uses the same ``seed`` as train so ``episode_train_mask`` picks one val
        set and val takes ``~train_mask`` (disjoint partition). Do not offset
        ``seed`` here — that would define a different val split and leak episodes
        into both train and val.

        Pass ``all_anchors=True`` when train also uses all_anchors so val sees
        every (segment, anchor) pair each epoch — deterministic and consistent
        with the training distribution.
        """
        return OptionAwareDataset(
            zarr_path=self.zarr_path,
            option_id_key=self.option_id_key,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            min_segment_len=self.min_segment_len,
            val_ratio=self._val_ratio,
            seed=self._seed,
            max_train_episodes=self._max_train_episodes,
            train=False,
            random_anchor=self.random_anchor,
            all_anchors=all_anchors,
            escape_relabeling=False,  # val always uses real labels
        )

    def __len__(self) -> int:
        if self._all_anchors:
            return len(self._samples)
        return len(self.segments)

    def _obs_window(self, seg: OptionSegment, anchor: int) -> dict[str, np.ndarray]:
        return obs_window_at_anchor(
            self._img, self._state, seg, anchor, self.n_obs_steps
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        from soda.dataset.option_stratified_sampler import decode_stratified_index

        is_escape = False
        if self._all_anchors:
            sample = self._samples[index]
            if len(sample) == 4:
                # escape_relabeling=True: 4-tuple (seg_idx, anchor, option_id_override, is_escape)
                seg_idx, anchor, option_id_out, is_escape = sample
            else:
                seg_idx, anchor = sample
                option_id_out = None  # filled below from seg
            seg = self.segments[seg_idx]
            if option_id_out is None:
                option_id_out = seg.option_id
        else:
            at_segment_end: bool | None = None
            if self._stratified_encoded_indices:
                seg_idx, at_segment_end = decode_stratified_index(index)
                seg = self.segments[seg_idx]
            else:
                seg = self.segments[index]

            if at_segment_end is True:
                anchor = seg.end - 1
            elif at_segment_end is False:
                anchor = int(self._rng.integers(seg.start, seg.end))
            elif self.random_anchor:
                anchor = int(self._rng.integers(seg.start, seg.end))
            else:
                anchor = seg.end - 1
            option_id_out = seg.option_id

        obs = self._obs_window(seg, anchor)
        actions_native = native_actions_from_anchor(self._action, seg, anchor)
        action = self._stretcher.stretch_with_duration(actions_native)
        # Escape examples always have beta=1; real examples use stored labels.
        beta = 1.0 if is_escape else float(self._beta_labels[anchor])
        remaining_len = int(seg.end - anchor)

        # Normalized progress within the segment [0, 1]: 0 at segment start, 1 at last frame.
        # Gives β a temporal position signal without requiring action history.
        seg_len = seg.length
        chunk_cursor = float(anchor - seg.start) / max(seg_len - 1, 1)
        chunk_cursor = min(max(chunk_cursor, 0.0), 1.0)

        return {
            "obs": {
                "image": torch.from_numpy(obs["image"]),
                "agent_pos": torch.from_numpy(obs["agent_pos"]),
            },
            "action": torch.from_numpy(action),
            "option_id": torch.tensor(option_id_out, dtype=torch.long),
            "beta_label": torch.tensor(beta, dtype=torch.float32),
            "is_escape": torch.tensor(is_escape, dtype=torch.bool),
            "chunk_cursor": torch.tensor(chunk_cursor, dtype=torch.float32),
            "segment_length": torch.tensor(remaining_len, dtype=torch.long),
            "segment_start": seg.start,
            "segment_end": seg.end,
            "anchor_index": anchor,
        }

    def get_beta_sample_indices(self) -> tuple[list[int], list[int]]:
        """Return ``(pos_indices, neg_indices)`` for :class:`AllAnchorsBetaStratifiedBatchSampler`.

        pos_indices: flat dataset indices where beta_label == 1 (segment-end anchors).
        neg_indices: flat dataset indices where beta_label == 0 (mid-segment anchors).
        Only valid when ``all_anchors=True``.
        """
        if not self._all_anchors:
            raise ValueError("get_beta_sample_indices requires all_anchors=True")
        pos: list[int] = []
        neg: list[int] = []
        for i, (_, anchor) in enumerate(self._samples):
            if self._beta_labels[anchor] == 1.0:
                pos.append(i)
            else:
                neg.append(i)
        return pos, neg

    def get_option_segment_bounds(self) -> list[tuple[int, int, int]]:
        """``(start, end, option_id)`` per segment for ``train_high`` / flow matching."""
        return [(s.start, s.end, s.option_id) for s in self.segments]

    def get_normalizer(
        self,
        mode: str = "limits",
        *,
        max_segments: int = 512,
        **kwargs: Any,
    ) -> Any:
        """
        DP-compatible ``LinearNormalizer`` for π_low training (requires ``diffusion_policy``).

        Fits ``action`` on stretched **D+1** suffix chunks (duration channel included),
        matching ``__getitem__`` (random anchor per segment). ``agent_pos`` / ``image``
        use the same rules as vanilla DP.
        """
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.model.common.normalizer import LinearNormalizer

        segments = self.segments
        if len(segments) > max_segments:
            rng = np.random.default_rng(0)
            idxs = rng.choice(len(segments), size=max_segments, replace=False)
            segments = [segments[int(i)] for i in idxs]

        fit_rng = np.random.default_rng(0)
        stretched: list[np.ndarray] = []
        for seg in segments:
            anchor = int(fit_rng.integers(seg.start, seg.end))
            native = native_actions_from_anchor(self._action, seg, anchor)
            stretched.append(self._stretcher.stretch_with_duration(native))
        actions = np.stack(stretched, axis=0).reshape(-1, stretched[0].shape[-1])

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


def build_option_dataset_from_config(cfg: Any) -> OptionAwareDataset:
    """
    Hydra-style factory: ``cfg.dataset`` (or flat ``cfg``) → ``OptionAwareDataset``.

    Expected keys: ``zarr_path``, ``option_id_key``, ``horizon``,
    ``n_obs_steps``, ``min_segment_len``, ``val_ratio``, ``seed``, ``max_train_episodes``.
    """
    ds_cfg = _config_get(cfg, "dataset", cfg)
    zarr_path = _config_get(ds_cfg, "zarr_path")
    if zarr_path is None:
        task = _config_get(ds_cfg, "task", "pusht")
        zarr_path = f"data/raw/{task}/{task}.zarr"

    horizon = _config_get(ds_cfg, "horizon", None)
    if horizon is None:
        horizon = _config_get(ds_cfg, "horizon_stretch_max", None)

    dataset = OptionAwareDataset(
        zarr_path=zarr_path,
        option_id_key=_config_get(ds_cfg, "option_id_key", "option_id_supervised"),
        horizon=horizon,
        n_obs_steps=int(_config_get(ds_cfg, "n_obs_steps", 2)),
        min_segment_len=int(_config_get(ds_cfg, "min_segment_len", 1)),
        val_ratio=float(_config_get(ds_cfg, "val_ratio", 0.0)),
        seed=int(_config_get(ds_cfg, "seed", 42)),
        max_train_episodes=_config_get(ds_cfg, "max_train_episodes", None),
        train=bool(_config_get(ds_cfg, "train", True)),
        random_anchor=bool(_config_get(ds_cfg, "random_anchor", True)),
        all_anchors=bool(_config_get(ds_cfg, "all_anchors", False)),
        escape_relabeling=bool(_config_get(ds_cfg, "escape_relabeling", False)),
    )
    return dataset
