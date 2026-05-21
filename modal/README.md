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
            → soda/eval/run_eval.py    # orchestrator
                 → policy_loaders      # dp_baseline | soda
                 → pusht_rollout.py
                 → metrics.py
```

**Every eval run requires `--checkpoint`.** `policy` only selects the loader:

| `policy` | Checkpoint | Loader |
|----------|------------|--------|
| `dp_baseline` | Columbia frozen, your retrained DP, etc. | `load_dp_image_policy` |
| `soda` | Trained SODA weights | `load_soda_policy` (TBD) |

## Commands (repo root)

| Command | Purpose |
|---------|---------|
| `modal run modal/modal_smoke.py` | Infra smoke (§7 row 5) |
| `modal run modal/modal_download_dp.py` | One-time download frozen DP to Volume |
| `modal run modal/modal_eval.py --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt` | DP smoke (5 eps) |
| `modal run modal/modal_eval.py --checkpoint .../latest.ckpt --full` | DP full (50 eps) |
| `modal run modal/modal_eval.py --policy soda --checkpoint /experiments/.../ckpt` | SODA (when implemented) |

Outputs: `/experiments/eval/<task>/<descriptive_run_name>/eval_log.json` on Volume **`soda-experiments`**.

Run folder name pattern:  
`{task}_{policy}_h{H}_{regime}_{smoke5|full50}_ckpt-{slug}_t{steps}_utc{timestamp}`  
(e.g. `pusht_diffusion_policy_h1_receding_smoke5_ckpt-frozen_latest_t300_utc20260522-153000`).

## Files

| File | Role |
|------|------|
| `modal_config.py` | `download_frozen_dp`, `eval_run`, `smoke`, `train_*` |
| `modal_download_dp.py` | CLI → download frozen Columbia weights |
| `modal_eval.py` | CLI → `eval_run.remote(...)` |

See [`configs/pusht/baseline_vanilla.yaml`](../configs/pusht/baseline_vanilla.yaml).
