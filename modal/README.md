# Modal (remote train/eval)

GPU work runs on [Modal](https://modal.com/), not on your laptop. See [`project_plan.md`](../project_plan.md) §3 Compute.

## Local setup

```powershell
conda activate soda
modal token new    # once
modal secret create wandb WANDB_API_KEY=<your_key>   # optional for smoke (uses soda-debug)
```

## Commands (from repo root)

| Command | Purpose |
|---------|---------|
| `modal run modal/modal_smoke.py` | **Smoke test** — GPU, torch, zarr, W&B, Volume (§7 row 5) |
| `modal run modal/modal_train_low.py` | Train π_low (when `soda/training/train_low.py` exists) |
| `modal run modal/modal_train_high.py` | Train π_high |
| `modal run modal/modal_eval.py` | Eval (stub) |

Examples:

```powershell
modal run modal/modal_smoke.py
modal run modal/modal_train_low.py --config-name soda_supervised --task pusht
```

## Files

| File | Role |
|------|------|
| `modal_config.py` | `app`, `image`, `volume`, `smoke`, `train_low`, `train_high` |
| `modal_smoke.py` | Local CLI → `smoke.remote()` |
| `modal_train_low.py` | Local CLI → `train_low.remote()` |
| `modal_train_high.py` | Local CLI → `train_high.remote()` |
| `modal_eval.py` | Local CLI → eval (stub) |
