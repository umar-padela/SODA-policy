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
modal run modal/modal_eval.py --checkpoint <path>   # local CLI
       → eval_run @ Modal GPU
            → soda/eval/run_eval.py
                 → dp_baseline → soda/eval/dp_runner.py
                 → soda         → soda/eval/soda_runner.py
                      (both → third_party PushTImageRunner + media/*.mp4)
                 → soda/eval/runner_common.py + metrics.py
```

**Every eval run requires `--checkpoint`.** `policy` only selects the loader:

| `policy` | Checkpoint | Backend |
|----------|------------|---------|
| `dp_baseline` | Columbia frozen, your retrained DP, etc. | `PushTImageRunner` (`n_action_steps` = `--action-horizon`, default **1**) |
| `soda` | Trained SODA weights | `pusht_rollout` (TBD) |

## Commands (repo root)

| Command | Purpose |
|---------|---------|
| `modal run modal/modal_smoke.py` | Infra smoke (§7 row 5) |
| `modal run modal/modal_download_dp.py` | One-time download frozen DP to Volume |
| `modal run modal/modal_eval.py --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt` | DP smoke (5 eps, videos for first 4) |
| `modal run modal/modal_eval.py --checkpoint .../latest.ckpt --n-test 1 --max-steps 300` | Single episode + full horizon |
| `modal run modal/modal_eval.py --checkpoint .../latest.ckpt --action-horizon 8` | DP paper-style receding (8 actions per replan) |
| `modal run modal/modal_eval.py --checkpoint .../latest.ckpt --full` | DP full (50 eps) |
| `modal run modal/modal_eval.py --checkpoint .../latest.ckpt --no-video` | Skip MP4 recording |

Outputs: `/experiments/eval/<task>/<descriptive_run_name>/` on Volume **`soda-experiments`**:

- `eval_log.json` — metrics + optional `dp_runner_log`
- `media/*.mp4` — rollout videos (first `n_test_vis` test episodes, default `min(n_test, 4)`)

Run folder name pattern:  
`{task}_{policy}_h{H}_{regime}_{smoke5|full50}_ckpt-{slug}_t{steps}_utc{timestamp}`

### Browse / download from Volume (local CLI)

The volume is mounted at `/experiments` **inside** Modal containers only. For `modal volume ls` / `get`, paths are **relative to the volume root** — do **not** include `/experiments`:

```powershell
modal volume ls soda-experiments
modal volume ls soda-experiments eval/pusht
modal volume get soda-experiments eval/pusht/<run_dir_name>/media/<id>.mp4 rollout.mp4
```

Example (from a completed eval):

```powershell
modal volume get soda-experiments eval/pusht/pusht_diffusion_policy_h1_receding_n1_ckpt-frozen_latest_t100_utc20260521-080952/media/6q5rjg5b.mp4 rollout.mp4
```

You can also browse files in the [Modal dashboard](https://modal.com/storage) → Volume `soda-experiments`.

## Files

| File | Role |
|------|------|
| `modal_config.py` | `download_frozen_dp`, `eval_run`, `smoke`, `train_*` |
| `modal_download_dp.py` | CLI → download frozen Columbia weights |
| `modal_eval.py` | CLI → `eval_run.remote(...)` |

See [`configs/pusht/baseline_vanilla.yaml`](../configs/pusht/baseline_vanilla.yaml).
