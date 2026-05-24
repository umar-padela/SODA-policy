# Option discovery

Offline pipelines that write option label arrays into task zarrs (`option_id_supervised`, `option_id_unsupervised`). See [`project_plan.md`](../../project_plan.md) §2 and §5.

| Subfolder | Role |
|-----------|------|
| [`supervised/`](supervised/README.md) | Heuristic (Push-T) + VLM placeholder (Square) |
| [`unsupervised/`](unsupervised/README.md) | LOVE adapter → unsupervised labels |

## Push-T supervised labels (E1)

```bash
# From repo root (requires zarr<3, numpy; Columbia zip — see supervised/pusht/README.md)
python -m soda.option_discovery.supervised.pusht.build_zarr --force   # first build
python -m soda.option_discovery.supervised.pusht.build_zarr             # relabel only if zarr exists

# Sanity-check labels (stats + segment histograms; frame + segment counts per skill)
python -m soda.option_discovery.supervised.pusht.visualize_labels
python -m soda.option_discovery.supervised.pusht.visualize_labels --save figures/pusht_labels.png --no-show

# Episode video → tmp/pusht_videos/ (gitignored)
python -m soda.option_discovery.supervised.pusht.play_episode --episode 100

# Run all checks (no GUI / no download)
python -m soda.option_discovery.supervised.pusht.check_pipeline
```

Writes `data/raw/pusht/pusht.zarr/` with `option_id_supervised`. If the zarr already exists, only relabels in place; `--force` rebuilds from Columbia zip (re-run LOVE afterward if needed).

Optional [Colab](https://colab.research.google.com/github/umar-padela/SODA-policy/blob/dev_umar/data/pusht/pushT_labeling.ipynb) for visualization / threshold checks (notebook not in repo; may 404 after branch cleanup).
