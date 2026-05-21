# SODA-policy

**SODA** — Supervised Option Discovery for Dynamic Action chunking.

Hierarchical imitation learning: a high-level policy selects **options**; a low-level **diffusion policy** outputs variable-horizon action chunks until a learned **termination** signal fires.

## Documentation

| Doc | Contents |
|-----|----------|
| [`project_plan.md`](project_plan.md) | **Main reference** — architecture, full repo tree, data policy, configs, execution runbook (§7) |
| [`project_proposal.md`](project_proposal.md) | Project proposal |
| Per-folder `README.md` | Short notes under `data/`, `soda/`, `soda/option_discovery/`, `configs/`, `scripts/`, `modal/`, `experiments/` |

## Repo layout (quick)

| Path | What |
|------|------|
| [`data/`](data/README.md) | Zarr datasets on disk |
| [`soda/option_discovery/`](soda/option_discovery/README.md) | Offline labeling (Push-T heuristic, LOVE, Square TBD) |
| [`data/raw/pusht/pusht.zarr/`](data/raw/pusht/pusht.zarr/) | Push-T training data (committed) |
| [`soda/`](soda/README.md) | Training and inference code |
| [`configs/`](configs/README.md) | Experiment YAML (`soda_supervised`, `soda_unsupervised`, baselines) |
| [`scripts/`](scripts/README.md) | Train / eval helpers |
| [`experiments/`](experiments/README.md) | Checkpoints and logs (gitignored) |
| [`third_party/`](third_party/) | Git submodules: [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [LOVE](https://github.com/yidingjiang/love) |

For every file and module role, see [`project_plan.md` §3–§4](project_plan.md).

## Setup

### 1. Clone (includes submodules)

`third_party/` is wired as git submodules. Use **`--recurse-submodules`** so Diffusion Policy and LOVE are checked out at the commits pinned in this repo:

```bash
git clone --recurse-submodules <repo-url>
cd SODA-policy
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

You do **not** need to run `scripts/setup_submodules.sh` for a normal clone — that script is only for the initial one-time `git submodule add` on the maintainer side.

Verify:

```bash
git submodule status
ls third_party/diffusion_policy
ls data/raw/pusht/pusht.zarr
```

### 2. Data

Push-T labeled data is **already in the repo** at:

```text
data/raw/pusht/pusht.zarr/
```

No download or manual copy step is required for training on Push-T.

To **regenerate** Push-T supervised labels (optional): `python -m soda.option_discovery.supervised.pusht.build_zarr` — see [`soda/option_discovery/README.md`](soda/option_discovery/README.md).

### 3. Environment

**Training runs on [Modal](https://modal.com/)** (GPU containers), not on your laptop. See [`project_plan.md` §3 Compute](project_plan.md).

| File | Use |
|------|-----|
| [`environment.yml`](environment.yml) | **Local conda env** — Modal, W&B, labeling tools (`zarr`, `opencv`, …) |
| [`environment.modal.yml`](environment.modal.yml) | Pin reference for Modal GPU image only (full DP stack) |
| `modal/modal_config.py` | Remote image + Volume for checkpoints |

```bash
conda env create -f environment.yml
conda activate soda
modal run modal/modal_smoke.py          # infrastructure smoke test
modal run modal/modal_download_dp.py    # one-time frozen DP ckpt on Volume
modal run modal/modal_eval.py --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt
# modal run modal/modal_train_low.py ...   (when train_low.py is implemented)
```

See [`project_plan.md` §7 row 5](project_plan.md).

### 4. Development

Implementation status and next steps: [`project_plan.md` §7 Execution runbook](project_plan.md).

```bash
# Example (once training is implemented)
# python soda/training/train_low.py --config-path configs/pusht --config-name soda_supervised
```

## Submodule pins

Submodule commits are fixed in the parent repo (see `git submodule status`). To bump a dependency, check out a new SHA inside the submodule, then commit the updated gitlink from the repo root — see [`project_plan.md` §7 row 7](project_plan.md).
