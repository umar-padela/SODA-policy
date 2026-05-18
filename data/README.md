# Data (files on disk)

This folder holds **datasets and notebooks**, not training code. PyTorch loaders live in [`soda/dataset/`](../soda/dataset/).

## Layout

| Path | Purpose |
|------|---------|
| `pusht/pushT_labeling.ipynb` | Colab notebook to regenerate Push-T labels |
| `square/square_labeling.ipynb` | Colab notebook for Square (later) |
| `raw/pusht/` | Place labeled Push-T zarr here (e.g. `pusht.zarr/`) — **you add this** |
| `raw/square/` | Square zarr (later) |
| `processed/` | Optional caches (gitignored) |

## Push-T zarr (E1)

After labeling, the dataset should live at:

```text
data/raw/pusht/pusht.zarr/
  data/img, state, action, option_id
  meta/episode_ends
```

Requires `zarr<3.0`.

## Regeneration (optional)

1. Open `pusht/pushT_labeling.ipynb` in Colab → **Run all**.
2. Upload the output zip to `data/raw/pusht/` and unzip to `pusht.zarr/`.
3. Commit updated zarr and/or notebook.

Day-to-day training reads the committed zarr; Colab is only for rebuilds.

See [`project_plan.md`](../project_plan.md) §5 for git policy.
