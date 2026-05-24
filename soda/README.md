# `soda` package

Python implementation of **SODA** (Supervised Option Discovery for Dynamic Action chunking): hierarchical imitation learning with offline option discovery, variable-horizon low-level chunks, and learned termination.

Design reference: [`project_plan.md`](../project_plan.md) §3–§4.

## Architecture (runtime)

```text
option_discovery/     offline → option_id_* in zarr
        ↓
dataset/                stretch segments, β labels, PyTorch batches
        ↓
training/               train π_high (FM) and π_low (diffusion + β BCE)
        ↓
models/                 LowPolicy, HighPolicy, TerminationHead
        ↓
inference/              HierarchicalPolicy (π_high + π_low + β loop)
        ↓
eval/                   Push-T sim, metrics, checkpoint loaders
```

## Subfolders

| Folder | README | Role |
|--------|--------|------|
| [`dataset/`](dataset/README.md) | ✓ | Zarr datasets, temporal stretch, segment indexing |
| [`models/`](models/README.md) | ✓ | Neural modules (π_high, π_low, termination) |
| [`training/`](training/README.md) | ✓ | Train loops, checkpoints, loss helpers |
| [`inference/`](inference/README.md) | ✓ | Hierarchical controller for sim / deploy |
| [`eval/`](eval/README.md) | ✓ | DP baseline + SODA eval, Modal entrypoints |
| [`option_discovery/`](option_discovery/README.md) | ✓ | Offline labeling (heuristic / LOVE / VLM) |

## Quick start

**Configs:** [`configs/README.md`](../configs/README.md) (parameter reference) + [`configs/pusht/`](../configs/pusht/) yaml files.

**Train (local or Modal):**

```bash
python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised
python soda/training/train_low.py  --config-path configs/pusht --config-name soda_supervised

modal run modal/modal_train_high.py --config-name soda_supervised
modal run modal/modal_train_low.py  --config-name soda_supervised
```

**Eval:**

```bash
# Frozen DP baseline
modal run modal/modal_eval.py --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt

# SODA (set inference.high_checkpoint / low_checkpoint in yaml first)
modal run modal/modal_eval.py --policy soda --checkpoint /path/to/best_low.ckpt
```

Requires `third_party/diffusion_policy` submodule and conda env from [`environment.yml`](../environment.yml).

## Naming cheatsheet

| Name | Meaning |
|------|---------|
| `horizon` | U-Net action prediction length (`null` → longest skill in zarr) |
| `n_action_steps` | π_low / eval: max actions per cached diffusion slice before forced replan |
| `n_obs_steps` | Observation history length (typically 2, matches DP) |
| `beta` / `β` | Termination logit from U-Net bottleneck at `t=0` |
| `option_balance` | `train_* .option_balance`: inverse-frequency **loss** weights per skill (default `inverse_freq`) |
| `termination_pos_weight` | BCE `pos_weight` on β=1 frames (auto γ ≈ 17 on Push-T); see `low_policy:` in yaml |

**Configs:** `soda_supervised.yaml` / `soda_unsupervised.yaml` (SODA), `dp_frozen.yaml` (frozen Columbia eval paths), `dp.yaml` (train self-hosted vanilla DP baseline).
