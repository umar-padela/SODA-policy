"""
Option-aware PyTorch dataset for SODA (project_plan §4.2, §7 row 11).

Extends Columbia Diffusion Policy zarr loading with:
- configurable ``option_id_key`` (supervised vs unsupervised column),
- per-frame ``beta_label`` via ``derive_beta_labels`` (§7 row 6 ☑),
- option-segment indexing and ``TemporalStretcher`` integration (row 10).

References
----------
- ``third_party/diffusion_policy/dataset/pusht_image_dataset.py`` — reuse patterns
- ``project_plan.md`` §5 — zarr layout; β not stored on disk

Implementation TODO (§7 row 11)
-------------------------------
- [ ] ``OptionLabeledZarrDataset.__init__``:
      - open zarr path from config (``data/raw/{task}/{task}.zarr``),
      - read ``option_id_key``, ``horizon`` / ``h_max``, image norm stats,
      - load ``episode_ends``, full ``option_ids`` column, precompute ``beta_label`` once,
      - build **segment index**: list of ``(start, end, option_id, episode_idx)`` per
        contiguous option run (do not cross episode boundaries).
- [ ] Decide **inherit vs wrap** DP ``PushTImageDataset`` (prefer wrap/subclass to avoid
      duplicating zarr I/O and observation normalization).
- [ ] ``__len__`` / ``__getitem__``:
      - sample a segment (or sliding window within segment — align with DP ``n_obs_steps``),
      - return obs dict compatible with DP (images, agent_pos, …),
      - return ``option_id`` (int ω), ``beta_label`` (float 0/1),
      - return native ``segment_length`` for stretcher / horizon target.
- [ ] Apply ``TemporalStretcher.stretch`` to action targets in ``__getitem__`` (or in
      collate) so batch actions are ``(h_max, D+1)`` with horizon channel filled.
- [ ] Optional: filter segments shorter than ``min_segment_len`` or longer than cap.
- [ ] ``get_option_segment_bounds()`` helper for ``train_high`` (flow matching on segment
      start states) — may live here or in a small util.
- [ ] Factory ``build_dataset(cfg)`` reading Hydra/OmegaConf YAML
      (``configs/pusht/soda_supervised.yaml``).
- [ ] Integration test: one batch from ``pusht.zarr`` shapes match DP + SODA keys;
      add ``tests/test_option_aware_dataset.py`` when implemented.

Done (§7 row 6)
---------------
- [x] ``derive_beta_labels(option_ids, episode_ends)`` — last frame per option segment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

# Version string for optional processed caches (bump if derivation rule changes).
DERIVE_BETA_LABELS_VERSION = "v1_last_frame_of_segment"


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

    Parameters
    ----------
    option_ids
        Shape ``(T,)`` integer option id per demo frame (e.g. from
        ``data/option_id_supervised``).
    episode_ends
        Shape ``(E,)`` cumulative exclusive end indices, same convention as
        Columbia DP / ``meta/episode_ends`` (episode ``e`` is
        ``option_ids[start:episode_ends[e]]`` with ``start=0`` or prior end).

    Returns
    -------
    np.ndarray
        Shape ``(T,)``, ``float32`` in ``{0.0, 1.0}``.

    Raises
    ------
    ValueError
        If shapes, monotonicity, or episode bounds are invalid.
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


class OptionLabeledZarrDataset:
    """
    PyTorch ``Dataset`` over a task zarr with option segments and β labels.

    Parameters
    ----------
    zarr_path
        e.g. ``data/raw/pusht/pusht.zarr``.
    option_id_key
        Zarr array name, e.g. ``option_id_supervised`` or ``option_id_unsupervised``.
    h_max
        Fixed action horizon after temporal stretching (DP U-Net horizon).
    """

    def __init__(
        self,
        zarr_path: str | Path,
        option_id_key: str = "option_id_supervised",
        h_max: int = 16,
        **kwargs: Any,
    ) -> None:
        self.zarr_path = Path(zarr_path)
        self.option_id_key = option_id_key
        self.h_max = h_max
        self._kwargs = kwargs
        raise NotImplementedError(
            "TODO §7 row 11: open zarr, build segment index, precompute beta_label"
        )

    def __len__(self) -> int:
        raise NotImplementedError("TODO §7 row 11")

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Return one training sample.

        Expected keys (align with DP + SODA training loop):
        - obs dict (image, state, …) — DP-compatible
        - ``action`` — stretched ``(h_max, D+1)`` with horizon channel
        - ``option_id`` — int ω
        - ``beta_label`` — float termination target
        - ``segment_length`` — native ``h_i`` before stretch (optional metadata)
        """
        raise NotImplementedError("TODO §7 row 11")


def build_option_dataset_from_config(cfg: Any) -> OptionLabeledZarrDataset:
    """Hydra-style factory: ``cfg.dataset`` → ``OptionLabeledZarrDataset``."""
    raise NotImplementedError("TODO §7 row 11: wire configs/pusht/soda_supervised.yaml")
