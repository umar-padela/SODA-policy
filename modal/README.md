# Modal (remote train/eval)

GPU work runs on [Modal](https://modal.com/), not on your laptop. See [`project_plan.md`](../project_plan.md) §3 Compute.

## Local setup

```powershell
conda activate soda
modal token new    # once
git submodule update --init --recursive third_party/diffusion_policy
```

## Eval architecture

```text
modal run modal/modal_eval.py --config configs/pusht/dp_frozen.yaml
       → eval_run @ Modal GPU
            → soda/eval/eval_yaml.py  (load yaml → EvalConfig)
            → soda/eval/run_eval.py
                 → dp_frozen / dp → soda/eval/dp_runner.py
                 → soda         → soda/eval/soda_runner.py
                      (both → third_party PushTImageRunner + media/*.mp4)
                 → soda/eval/runner_common.py + metrics.py
```

**Every eval run uses `--config` pointing at a yaml file** (`dp_frozen.yaml`, `soda_supervised.yaml`, …).
Checkpoint paths and eval settings live in that yaml; CLI flags override.

| Config | Policy | Checkpoint source in yaml |
|--------|--------|----------------------------|
| `dp_frozen.yaml` | `eval.policy: dp_frozen` | `checkpoint.volume_path` |
| `dp.yaml` | `eval.policy: dp` | `checkpoint.volume_path` (self-trained) |
| `soda_supervised.yaml` | `eval.policy: soda` | `inference.high_checkpoint` + `inference.low_checkpoint` |

## Commands (repo root)

| Command | Purpose |
|---------|---------|
| `modal run modal/modal_smoke.py` | Infra smoke (§7 row 5) |
| `modal run modal/modal_download_dp.py` | One-time download frozen DP to Volume |
| `modal run modal/modal_eval.py --config configs/pusht/dp_frozen.yaml` | DP smoke eval (settings from yaml) |
| `modal run modal/modal_eval.py --config configs/pusht/dp_frozen.yaml --n-test 1` | Single episode + yaml max_steps |
| `modal run modal/modal_eval.py --config configs/pusht/dp_frozen.yaml --full` | DP full (50 eps from yaml) |
| `modal run modal/modal_eval.py --config configs/pusht/soda_supervised.yaml` | SODA eval (paths in yaml `inference.*`) |
| `modal run modal/modal_rollout_low_policy.py --run-readme "..." --episode 100 --option-id 0` | Expert-anchored π_low segment rollout + side-by-side MP4 |
| `modal run modal/modal_train_low.py` | Train π_low |
| `modal run modal/modal_train_high.py --low-checkpoint ...` | Train π_high |
| `modal run modal/modal_train_dp.py` | Train vanilla DP from `configs/pusht/dp.yaml` |
| `modal run modal/modal_eval.py --config configs/pusht/dp.yaml` | Eval self-trained DP |

Outputs: `/experiments/eval/<config_stem>/<YYYYMMDD>/<HHMMSS>/` on Volume **`soda-experiments`**:

- `config.yaml` — exact yaml snapshot used for the run
- `run_manifest.json` — invoke command, CLI overrides, resolved eval settings
- `command.txt` — same invoke command as plain text
- `eval_log.json` — metrics + optional runner log
- `media/*.mp4` — rollout videos named `{config_stem}_ep{idx}_seed{seed}_score{pct}.mp4`

Example layout: `eval/dp_frozen/20260525/014851/`

Legacy descriptive name (policy, horizon, checkpoint slug) is stored in `eval_log.json` as `descriptive_run_name`, not in the folder path.

### Browse / download from Volume (local CLI)

The volume is mounted at `/experiments` **inside** Modal containers only. For `modal volume ls` / `get`, paths are **relative to the volume root** — do **not** include `/experiments`:

```powershell
modal volume ls soda-experiments
modal volume ls soda-experiments eval/dp_frozen
modal volume get soda-experiments eval/dp_frozen/20260525/014851/media/dp_frozen_ep00_seed10000_score055.mp4 rollout.mp4
modal volume get soda-experiments segment_rollout/seg0001_ep100_o0_reposition_anchor12345.mp4 rollout.mp4
```

You can also browse files in the [Modal dashboard](https://modal.com/storage) → Volume `soda-experiments`.

Long train/eval jobs use ``Function.spawn()`` instead of ``.remote()`` so work continues
if the local ``modal run`` client disconnects. Each run prints a ``FunctionCall`` object id;
monitor progress in the [Modal dashboard](https://modal.com/apps).

## GPUs

| Function | Default GPU | Override env var |
|----------|-------------|------------------|
| `smoke`, `download_frozen_dp`, `eval_run` | **A10G** | `MODAL_GPU_EVAL` |
| `train_low`, `train_high` | **L40S** | `MODAL_GPU_TRAIN` |

Example (Windows): `$env:MODAL_GPU_TRAIN="A100-40GB"; modal run modal/modal_train_low.py`

## Files

| File | Role |
|------|------|
| `modal_config.py` | `download_frozen_dp`, `eval_run`, `smoke`, `train_*` |
| `modal_download_dp.py` | CLI → download frozen Columbia weights |
| `modal_eval.py` | CLI → `eval_run` via `spawn()` |
| `modal_train_low.py` / `modal_train_high.py` | CLI → `train_*` via `spawn()` (`--detach` to not wait) |

See [`configs/pusht/dp_frozen.yaml`](../configs/pusht/dp_frozen.yaml).
