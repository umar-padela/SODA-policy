"""Push-T zarr → fixed-length state/action windows for LOVE.

Two Datasets:
- `PushtLoveDataset` — sliding windows of length `window_len` for training.
- `PushtFullEpisodeDataset` — one episode per item for inference/labeling.

Both read state and action from `data/raw/pusht/pusht.zarr` (rotation stays
in radians as stored — LOVE doesn't care about units). Actions are
discretized to `num_action_bins` cluster ids because `hssm_rl.EnvModel`
reconstructs actions with `F.cross_entropy` and only accepts integer ids;
see `quantize.py` for the codebook fit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from soda.option_discovery.unsupervised.love_adapter.quantize import (
    fit_kmeans,
    quantize,
)


def _read_arrays(zarr_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = zarr.open(str(zarr_path), mode="r")
    state = np.asarray(root["data"]["state"][:], dtype=np.float32)
    action = np.asarray(root["data"]["action"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return state, action, episode_ends


class PushtLoveDataset(Dataset):
    """Sliding fixed-length windows over Push-T episodes, with quantized actions."""

    def __init__(
        self,
        zarr_path: Path,
        window_len: int,
        num_action_bins: int,
        stride: int | None = None,
        action_centroids: np.ndarray | None = None,
        kmeans_seed: int = 0,
    ):
        self.window_len = window_len
        self.stride = stride or max(1, window_len // 2)

        state, action, episode_ends = _read_arrays(zarr_path)
        self.state = state
        self.action_continuous = action

        if action_centroids is None:
            self.action_centroids = fit_kmeans(
                action, n_clusters=num_action_bins, seed=kmeans_seed
            )
        else:
            self.action_centroids = action_centroids.astype(np.float32)

        self.action_ids = quantize(action, self.action_centroids)

        starts = np.concatenate([[0], episode_ends[:-1]])
        self.windows: list[tuple[int, int]] = []
        for s, e in zip(starts, episode_ends):
            last_start = e - window_len + 1
            for t in range(s, last_start, self.stride):
                self.windows.append((t, t + window_len))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        a, b = self.windows[idx]
        obs = torch.from_numpy(self.state[a:b])              # (window_len, 5) float32
        act = torch.from_numpy(self.action_ids[a:b]).long()  # (window_len,) int64
        return obs, act


class PushtFullEpisodeDataset(Dataset):
    """One item per episode, full length. Used at labeling time."""

    def __init__(
        self,
        zarr_path: Path,
        action_centroids: np.ndarray,
    ):
        state, action, episode_ends = _read_arrays(zarr_path)
        self.state = state
        self.action_centroids = action_centroids.astype(np.float32)
        self.action_ids = quantize(action, self.action_centroids)
        starts = np.concatenate([[0], episode_ends[:-1]])
        self.spans = list(zip(starts.tolist(), episode_ends.tolist()))

    def __len__(self) -> int:
        return len(self.spans)

    def __getitem__(self, idx: int):
        a, b = self.spans[idx]
        obs = torch.from_numpy(self.state[a:b])
        act = torch.from_numpy(self.action_ids[a:b]).long()
        return obs, act, int(a), int(b)
