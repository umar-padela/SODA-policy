# `soda/dataset`

PyTorch data loading for SODA training: read DP-format zarr, index **option segments**, stretch variable-length skills to a fixed horizon, and derive termination labels.

Columbia layout: `data/raw/pusht/pusht.zarr` (`img`, `state`, `action`, `option_id_*`, `meta/episode_ends`).

## Files

| File | Purpose |
|------|---------|
| [`option_aware_dataset.py`](option_aware_dataset.py) | `OptionAwareDataset` — π_low training samples |
| [`option_start_dataset.py`](option_start_dataset.py) | `OptionStartDataset` — π_high segment-start samples |
| [`temporal_stretch.py`](temporal_stretch.py) | `TemporalStretcher` — linear resample + duration channel |

## `OptionStartDataset` (π_high)

**One sample** = one option segment at **segment start**:

- `obs` — `(n_obs_steps, …)` image + agent_pos window ending at `segment.start`
- `option_id` — discrete ω the expert chose for this skill

Factory: `build_option_start_dataset_from_config(cfg)`.

## `OptionAwareDataset` (π_low)

**One sample** = one option segment with:

- `obs` — `(n_obs_steps, …)` image + agent_pos window ending at random `anchor` (train) or last frame (val)
- `action` — `(horizon, D+1)` stretched **suffix** `action[anchor:seg.end]`; last channel = normalized remaining duration
- `option_id` — discrete ω for this segment
- `beta_label` — 1 only when `anchor` is the segment's last frame; else 0

Factory: `build_option_dataset_from_config(cfg)` reads Hydra `task.dataset` block plus root `horizon`.

**Normalizer:** `get_normalizer()` fits on stretched D+1 suffix chunks (random anchor per segment, up to 512 samples), used by `train_low`.

## Segment indexing

`build_option_segment_index(option_ids, episode_ends)` → list of `OptionSegment`:

```python
OptionSegment(start, end, option_id, episode_idx)
```

Contiguous runs of the same `option_id` within each episode. Used by both π_high and π_low datasets.

## `derive_beta_labels`

`beta_label = 1` on the **last frame** of each option segment; 0 elsewhere. Computed at load time (§8 locked — not written to zarr). On Push-T E1 there is one positive frame per segment (~**5.6%** of all frames); `train_low` auto-sets BCE `pos_weight` ≈ mean segment length − 1.

## Class / label expectations

Push-T supervised (E1): **`num_options = 3`**, contiguous ids **0–2** (REPOSITION, LINEAR-PUSH, PIVOT-PUSH). Regenerate labels after heuristic changes — see [`option_discovery/supervised/pusht/README.md`](../option_discovery/supervised/pusht/README.md).

## Temporal stretch

Native segment length `h_orig` → fixed `horizon` via linear interpolation on actions.

- **Duration channel:** `h_orig / horizon` broadcast to all timesteps
- **At inference:** mean-pool duration channel × `horizon` → decode native length (`decode_segment_steps`)

Root yaml `horizon: null` → auto-set to longest option segment in the dataset.

## Config keys

| Key | Description |
|-----|-------------|
| `horizon` (root) | U-Net prediction length; `null` = longest segment |
| `task.dataset.zarr_path` | Path to task zarr |
| `task.dataset.option_id_key` | `option_id_supervised` or `option_id_unsupervised` |
| `n_obs_steps` | Usually from root yaml |
| `val_ratio`, `seed` | Episode-level train/val split |
| `min_segment_len` | Drop very short segments |
| `random_anchor` | π_low: random frame within segment when not using stratified encoded indices |

## Tests

```bash
pytest tests/test_option_aware_dataset.py tests/test_option_start_dataset.py tests/test_temporal_stretch.py tests/test_derive_beta_labels.py
```

## See also

- [`option_discovery/README.md`](../option_discovery/README.md) — how labels get into zarr
- [`training/README.md`](../training/README.md) — consumes these datasets
