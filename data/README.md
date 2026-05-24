# Data (files on disk)

This folder holds **datasets** (zarr stores), not training code. PyTorch loaders live in [`soda/dataset/`](../soda/dataset/). Option labeling scripts live in [`soda/option_discovery/`](../soda/option_discovery/).

## Layout

| Path | Purpose |
|------|---------|
| `raw/pusht/pusht.zarr/` | Push-T demos + option label arrays (in git) |
| `raw/square/square.zarr/` | Square zarr (later) |
| `processed/` | Optional caches (gitignored), e.g. Columbia replay zip for regen |

## Push-T zarr

One zarr per task. Demonstrations plus option label arrays (see [`project_plan.md`](../project_plan.md) §5):

```text
data/raw/pusht/pusht.zarr/
  data/img, state, action
  data/option_id_supervised      # E1 heuristic (build script below)
  data/option_id_unsupervised    # E3 LOVE (when pipeline runs)
  meta/episode_ends
```

Configs set `option_id_key` to select supervised vs unsupervised column. Push-T E1 supervised labels use **3 skills** (ids **0–2**); see [`soda/option_discovery/supervised/pusht/README.md`](../soda/option_discovery/supervised/pusht/README.md).

Requires `zarr<3.0`.

## Regenerate Push-T supervised labels

From repo root:

```bash
python -m soda.option_discovery.supervised.pusht.build_zarr
```

Place the Columbia replay zip at `data/processed/pusht_cache/pusht_cchi_v7_replay.zarr.zip` (see [`soda/option_discovery/supervised/pusht/README.md`](../soda/option_discovery/supervised/pusht/README.md)), then `build_zarr --force` for a full store. If `pusht.zarr` already exists, running without `--force` only updates `option_id_supervised`.

**Check labels:** `python -m soda.option_discovery.supervised.pusht.visualize_labels`

Day-to-day training reads the committed zarr; rebuild only when thresholds or export logic change.

See [`project_plan.md`](../project_plan.md) §5 for git policy.
