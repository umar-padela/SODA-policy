# `soda/training`

Training entrypoints and loss helpers for π_high and π_low. Config-driven via Hydra yaml under [`configs/pusht/`](../../configs/pusht/).

## Files

| File | Purpose |
|------|---------|
| [`train_high.py`](train_high.py) | π_high FM training loop |
| [`train_low.py`](train_low.py) | π_low diffusion + termination training |
| [`losses_low.py`](losses_low.py) | `diffusion_loss`, `termination_bce_loss`, `low_policy_total_loss` |
| [`option_balance.py`](option_balance.py) | Inverse-frequency skill weights + auto `termination_pos_weight` (γ) |

π_high flow-matching loss lives in `HighPolicy.compute_loss` (no separate `losses_high.py`).

## π_high — `train_high.py`

**Data:** [`OptionStartDataset`](../dataset/option_start_dataset.py) — one sample per option segment at **segment start** (obs window ending at `segment.start`).

**Policy build:** frozen vision from `train_high.low_checkpoint` (π_low) or `train_high.dp_checkpoint` (Columbia DP) + trainable FM head + option embedding table.

**Recommended order:** train π_low → set `train_high.low_checkpoint` → train π_high.

**Checkpoints** (`best.ckpt`, `latest.ckpt`, `epoch_*.ckpt`):

```python
{
  "policy": state_dict,
  "high_policy_config": {...},
  "normalizer": obs_encoder normalizer,
  "cfg": full Hydra config,
  "epoch", "metrics", ...
}
```

**Run:**

```bash
python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised
modal run modal/modal_train_high.py --config-name soda_supervised
```

Requires a vision checkpoint: `train_high.low_checkpoint` (trained π_low) or `train_high.dp_checkpoint` (Columbia DP defaults).

**Class balancing:** `train_high.option_balance: inverse_freq` (default in `soda_supervised.yaml`) scales each segment’s FM loss by inverse skill frequency. Validation stays unweighted. Disable with `train_high.option_balance=none`.

## π_low — `train_low.py`

**Data:** `OptionAwareDataset` — stretched D+1 action chunks, random anchor within segment (train).

**Policy build:** `LowPolicy` with normalizer from `train_dataset.get_normalizer()`.

Optional init: `train_low.finetune_dp_checkpoint` → load DP weights with `strict=False`.

At startup, `train_low` prints **option inverse-frequency weights** (if enabled) and **termination BCE `pos_weight`** (γ ≈ train β=0 frames / β=1 frames). Both apply to **train loss only**; val uses unweighted option sampling and the same γ from config (for comparable loss scale).

**Checkpoints:**

```python
{
  "policy": state_dict,
  "normalizer": LinearNormalizer state,
  "low_policy_config": {...},      # includes termination_pos_weight
  "horizon": int,
  "cfg": full Hydra config,
  ...
}
```

**Run:**

```bash
python soda/training/train_low.py --config-path configs/pusht --config-name soda_supervised
modal run modal/modal_train_low.py --config-name soda_supervised
```

## Losses — `losses_low.py`

| Function | Gradients into |
|----------|----------------|
| `diffusion_loss` | U-Net, obs encoder, option embed |
| `termination_bce_loss` | `TerminationHead` only (bottleneck detached); optional `pos_weight` for β=1 |
| `low_policy_total_loss` | weighted sum |

Wiring: `LowPolicy.compute_loss(batch, class_weights=…)` — option `inverse_freq` weights and `termination_pos_weight` (γ) are resolved in `train_low.py` / `train_high.py` and stored in `LowPolicyConfig`.

## Config blocks (yaml)

**Full parameter reference (what to lock before train):** [`configs/README.md`](../../configs/README.md).

| Block | Used by |
|-------|---------|
| `high_policy:` | π_high architecture |
| `train_high:` | LR, epochs, dp_checkpoint, `option_balance` |
| `low_policy:` | option embed, termination head, `termination_pos_weight` |
| `policy:` | U-Net / diffusion inference steps (π_low) |
| `noise_scheduler:` | DDPM schedule |
| `train_low:` | LR, epochs, finetune_dp_checkpoint, `option_balance` |
| `task.dataset:` | zarr path, option_id_key (shared) |

## Output directories

Default (local):

- `experiments/train_high/{name}/`
- `experiments/train_low/{name}/`

Modal: `/experiments/train_high/{task}_{config_name}/` on Volume.

## Tests

```bash
pytest tests/test_train_low_scaffold.py tests/test_high_policy.py tests/test_losses_low.py tests/test_option_balance.py
```

## See also

- [`eval/policy_loaders.py`](../eval/policy_loaders.py) — reload checkpoints for sim eval
- [`dataset/README.md`](../dataset/README.md) — training data layout
