# Push-T supervised labeling (E1)

Heuristic option labels → `data/raw/pusht/pusht.zarr` (`option_id_supervised`).

Skills (contiguous ids **0–2**): **REPOSITION**, **LINEAR-PUSH**, **PIVOT-PUSH**. Training expects all three ids present (`num_options=3`).

## 1. Download Columbia replay zip (once)

Browser: https://drive.google.com/uc?id=1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t

Save as:

```text
data/processed/pusht_cache/pusht_cchi_v7_replay.zarr.zip
```

(`data/processed/` is gitignored.)

## 2. Build zarr (repo root, `conda activate soda`)

```bash
# First build (needs zip above):
python -m soda.option_discovery.supervised.pusht.build_zarr --force

# Later: refresh labels only (no zip needed if pusht.zarr exists):
python -m soda.option_discovery.supervised.pusht.build_zarr

# Full rebuild from zip (wipes zarr, including option_id_unsupervised):
python -m soda.option_discovery.supervised.pusht.build_zarr --force
```

## 3. Visualize / check

```bash
python -m soda.option_discovery.supervised.pusht.visualize_labels
python -m soda.option_discovery.supervised.pusht.visualize_labels --save data/processed/pusht_labels.png --no-show

python -m soda.option_discovery.supervised.pusht.play_episode --episode 100
python -m soda.option_discovery.supervised.pusht.play_episode --episode 0 --open

python -m soda.option_discovery.supervised.pusht.check_pipeline
```

`visualize_labels` prints **frame counts** and **training segment counts** per skill (one segment = one π_high / π_low sample). `check_pipeline` verifies ids `{0,1,2}`, relabel consistency, and smoke-tests video export.

`build_zarr` validates contiguous skill ids after relabeling.

Optional [Colab](https://colab.research.google.com/github/umar-padela/SODA-policy/blob/dev_umar/data/pusht/pushT_labeling.ipynb) for interactive threshold tuning (not in repo).
