"""
State-based (lowdim) variant of OptionAwareDataset for block push training.

Returns ``obs`` as (n_obs_steps, state_dim) float32 tensor instead of the
image/agent_pos dict that the image-based dataset returns. Everything else
(option_id, beta_label, action, stretching) is identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from soda.dataset.option_aware_dataset import (
    OptionAwareDataset,
    OptionSegment,
    native_actions_from_anchor,
)


class OptionAwareLowdimDataset(OptionAwareDataset):
    """State-only option-aware dataset.  ``obs_dict['obs']`` is a flat state window."""

    def _obs_window(self, seg: OptionSegment, anchor: int) -> np.ndarray:
        """Return (n_obs_steps, state_dim) float32 — no image."""
        win_end = anchor + 1
        win_start = max(seg.start, win_end - self.n_obs_steps)

        states: list[np.ndarray] = []
        for idx in range(win_start, win_end):
            states.append(np.asarray(self._state[idx], dtype=np.float32))

        # Pad at the beginning by repeating the first frame
        while len(states) < self.n_obs_steps:
            states.insert(0, states[0].copy())

        return np.stack(states[-self.n_obs_steps:], axis=0)   # (n_obs_steps, state_dim)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from soda.dataset.option_stratified_sampler import decode_stratified_index

        is_escape = False
        if self._all_anchors:
            sample = self._samples[index]
            if len(sample) == 4:
                seg_idx, anchor, option_id_out, is_escape = sample
            else:
                seg_idx, anchor = sample
                option_id_out = None
            seg = self.segments[seg_idx]
            if option_id_out is None:
                option_id_out = seg.option_id
        else:
            at_segment_end = None
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

        obs_np = self._obs_window(seg, anchor)          # (n_obs_steps, state_dim)
        actions_native = native_actions_from_anchor(self._action, seg, anchor)
        action = self._stretcher.stretch_with_duration(actions_native)
        beta = 1.0 if is_escape else float(self._beta_labels[anchor])
        remaining_len = int(seg.end - anchor)

        seg_len = seg.length
        chunk_cursor = float(anchor - seg.start) / max(seg_len - 1, 1)
        chunk_cursor = min(max(chunk_cursor, 0.0), 1.0)

        return {
            "obs": {"obs": torch.from_numpy(obs_np)},
            "action": torch.from_numpy(action),
            "option_id": torch.tensor(option_id_out, dtype=torch.long),
            "beta_label": torch.tensor(beta, dtype=torch.float32),
            "is_escape": torch.tensor(is_escape, dtype=torch.bool),
            "chunk_cursor": torch.tensor(chunk_cursor, dtype=torch.float32),
            "segment_length": torch.tensor(remaining_len, dtype=torch.long),
            "segment_start": seg.start,
            "segment_end": seg.end,
        }

    def get_validation_dataset(self, *, all_anchors: bool = False) -> "OptionAwareLowdimDataset":
        return OptionAwareLowdimDataset(
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
            escape_relabeling=False,
        )
