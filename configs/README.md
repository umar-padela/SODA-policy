# Configs

Hydra YAML recipes per **task** and **supervision type**. This is where you **lock model architecture before training**.

**Source of truth:** [`pusht/soda_supervised.yaml`](pusht/soda_supervised.yaml) (and `soda_unsupervised.yaml`). Dataclass definitions: `HighPolicyConfig` in [`soda/models/high_policy.py`](../soda/models/high_policy.py), `LowPolicyConfig` in [`soda/models/low_policy.py`](../soda/models/low_policy.py).

```bash
python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised
python soda/training/train_low.py  --config-path configs/pusht --config-name soda_supervised
```

## Naming

| Config | `pusht/` | `square/` |
|--------|----------|-----------|
| `soda_supervised.yaml` | E1 — `option_id_supervised` | E2 — VLM (TBD) |
| `soda_unsupervised.yaml` | E3 — `option_id_unsupervised` | E4 — LOVE |
| `dp_frozen.yaml` | Frozen Columbia DP checkpoint paths | Eval baseline only |
| `dp.yaml` | Self-trained vanilla DP (Columbia hyperparams) | Train + eval via `modal_train_dp.py` / `modal_eval.py` |

**Vanilla DP (not SODA):** `dp.yaml` is fully wired — every key is tagged `[train]`, `[eval]`, or `[train+eval]` inline. Training uses Columbia workspace via `soda/training/train_dp.py`; eval uses the same file.

**Tag legend:** `[train]` = training only · `[eval]` = inference/rollout only · `[train+eval]` = both (e.g. architecture).

---

## Shared (root yaml) — both π_high and π_low

These live at the **top level** of `soda_*.yaml`. Changing them after training requires retraining (or careful checkpoint surgery).

| Key | Example | Locks in | Notes |
|-----|---------|----------|-------|
| `n_obs_steps` | `2` | Both policies, env | Observation history length (DP convention) |
| `n_action_steps` | `8` | **eval only** | Receding-horizon execute window before replan (h=8); not a training hyperparameter |
| `horizon` | `null` / `16` | train+eval | U-Net chunk length (16 for DP; auto-stretch for SODA) |
| `past_action_visible` | `false` | Env / policy | Push-T: keep false |
| `obs_as_global_cond` | `true` | π_low | Must be `true`; `false` raises (inpainting not supported) |

### `task.dataset` (data + stretch)

| Key | Example | Locks in | Notes |
|-----|---------|----------|-------|
| `zarr_path` | `data/raw/pusht/pusht.zarr` | Training data | |
| `option_id_key` | `option_id_supervised` | Label column | `_supervised` vs `_unsupervised` |
| `val_ratio` | `0.1` | Split only | Episode-level train/val |
| `seed` | `42` | Split only | |
| `min_segment_len` | `1` | Dataset filter | Drop tiny segments |
| `num_workers` | `4` | DataLoader | |

### `task.env_runner` (eval only)

Used by sim eval / runners, not model weights: `max_steps`, `legacy_test`, `test_start_seed`, video settings.

---

## π_high — `high_policy:` + `train_high:`

Maps to **`HighPolicyConfig`** + frozen DP vision.

### `high_policy:` (architecture you must lock)

| Key | Example | Required | Meaning |
|-----|---------|----------|---------|
| `num_options` | `null` | auto if null | \|unique option ids\| in zarr (Push-T E1: **3** — ids **0–2**); set explicitly to override |
| `option_embed_dim` | `32` | **yes** | FM target dim; `nn.Embedding` width |
| `global_feat_dim` | `null` | auto if null | From frozen DP encoder: `obs_feature_dim × n_obs_steps`; must match encoder |
| `fm_hidden_dim` | `256` | **yes** | Option velocity MLP hidden size |
| `fm_num_layers` | `3` | **yes** | Velocity MLP depth |
| `time_embed_dim` | `32` | **yes** | Sinusoidal time embed (even, ≥ 4) |
| `num_inference_steps` | `10` | **yes** | Euler steps for `sample_option` at inference |

**Frozen at train time (not in `high_policy:`):** vision encoder + normalizer from either:

- **`train_high.low_checkpoint`** — trained π_low `best.ckpt` (preferred; train π_low first), or
- **`train_high.dp_checkpoint`** — Columbia frozen DP `.ckpt` (fallback if `low_checkpoint` is null)

### `train_high:` (optimization — does not change architecture)

| Key | Example | Notes |
|-----|---------|-------|
| `device` | `cuda:0` | |
| `seed` | `42` | |
| `num_epochs` | `100` | |
| `batch_size` | `64` | |
| `lr` | `1e-4` | |
| `weight_decay` | `1e-6` | |
| `checkpoint_every` | `10` | |
| `output_dir` | `null` | Default `experiments/train_high/{name}/` |
| `low_checkpoint` | path | **Preferred:** π_low `.ckpt` → frozen `ObsEncoder` (ResNet from trained π_low) |
| `dp_checkpoint` | `null` | Fallback Columbia DP `.ckpt` if `low_checkpoint` null; null = search default paths |
| `wandb_*` | | Logging |
| `option_balance` | `inverse_freq` | `none` \| `inverse_freq` — train loss weights per skill segment count (val unweighted) |

**Class imbalance (π_high):** Push-T E1 has ~50% REPOSITION / ~25% LINEAR / ~25% PIVOT segment starts. `inverse_freq` scales each sample’s FM loss by `N / (K × count[ω])` so gradient contribution is balanced without changing the dataloader. Use `option_balance=none` for a natural-frequency baseline.

---

## π_low — `low_policy:` + `policy:` + `noise_scheduler:` + `train_low:`

Maps to **`LowPolicyConfig`** + DP hybrid backbone kwargs.

### `low_policy:` (SODA-specific — lock before train)

| Key | Example | Required | Meaning |
|-----|---------|----------|---------|
| `num_options` | `null` | auto if null | Must match π_high option count (Push-T E1: **3**) |
| `option_embed_dim` | `32` | **yes** | Concatenated to U-Net `global_cond` |
| `bottleneck_dim` | `2048` | **yes** | U-Net mid channel dim; must match `policy.down_dims[-1]` |
| `termination_loss_weight` | `1.0` | **yes** | λ for BCE vs diffusion MSE; use **`0.1`** if `termination_stop_grad=false` |
| `termination_pos_weight` | `null` | auto if null | PyTorch BCE `pos_weight` on `beta_label=1`; auto = train β=0 frames / β=1 frames (~**17** on Push-T) |
| `termination_input` | `bottleneck` | **yes** | `bottleneck` (U-Net t=0 plan) or `obs` (`global_cond` only) |
| `termination_stop_grad` | `true` | **yes** | `true` = §8 (β MLP only); `false` = β grads into U-Net t=0 path / encoder |
| `termination_head.bottleneck_dim` | `2048` | **yes** | Usually same as `bottleneck_dim` |
| `termination_head.hidden_dim` | `256` | **yes** | β MLP hidden |
| `termination_head.num_layers` | `2` | **yes** | β MLP depth (≥ 1) |

**Termination imbalance:** One `beta_label=1` frame per option segment end vs ~17 negatives per segment on average. Auto `termination_pos_weight` upweights β=1 rows in BCE so the head does not collapse to always predicting 0. Tune `inference.beta_transition` if termination fires too early at rollout.

**From dataset (not in `low_policy:`):** root `horizon` → `LowPolicy.horizon` (U-Net sequence length; auto-resolved when `null`).

### `policy:` (DP hybrid U-Net + vision — lock before train)

Passed to `LowPolicy(...)` constructor (Columbia `DiffusionUnetHybridImagePolicy`).

| Key | Example | Meaning |
|-----|---------|---------|
| `crop_shape` | `[84, 84]` | Image crop for hybrid encoder (Columbia Push-T) |
| `num_inference_steps` | `100` | DDPM denoising steps at inference |
| `diffusion_step_embed_dim` | `128` | U-Net time embedding |
| `down_dims` | `[512, 1024, 2048]` | U-Net channel pyramid; **last = `bottleneck_dim`** |
| `kernel_size` | `5` | U-Net conv kernel |
| `n_groups` | `8` | GroupNorm groups |
| `cond_predict_scale` | `true` | FiLM scale in U-Net |
| `eval_fixed_crop` | `true` | Fixed crop at eval (Columbia) |
| `obs_encoder_group_norm` | `true` | Vision backbone GN (Columbia) |

Also wired from root yaml: `n_obs_steps`, `n_action_steps`, `horizon` (U-Net prediction length).

### `noise_scheduler:` (DDPM — lock before train)

| Key | Example |
|-----|---------|
| `num_train_timesteps` | `100` |
| `beta_start` / `beta_end` | `1e-4` / `0.02` |
| `beta_schedule` | `squaredcos_cap_v2` |
| `clip_sample` | `true` |
| `prediction_type` | `epsilon` |
| `variance_type` | `fixed_small` |

### `train_low:` (optimization + optional init)

| Key | Example | Notes |
|-----|---------|-------|
| `finetune_dp_checkpoint` | `null` | Optional DP warm-start (`strict=False` load) |
| `option_balance` | `inverse_freq` | Same as `train_high.option_balance` — per-skill diffusion + termination loss scaling (val unweighted) |
| (others) | | Same pattern as `train_high:` (`device`, `lr`, `epochs`, …) |

---

## Inference — `inference:` (eval only, not training architecture)

| Key | Example | Meaning |
|-----|---------|---------|
| `beta_transition` | `0.5` | Segment ends when β > threshold |
| `beta_diffusion_t` | `0` | Diffusion t for cheap β forward |
| `high_checkpoint` | path | π_high `.ckpt` for sim eval |
| `low_checkpoint` | path | π_low `.ckpt` for sim eval |

---

## Pre-train checklist

Before starting a run you intend to keep:

1. **`horizon`** — longest skill must fit; sets π_low U-Net tensor shape (`null` = auto from zarr).
2. **`num_options`** — consistent across π_high / π_low (auto from zarr; Push-T E1 must be **3** contiguous ids **0–2**).
3. **`option_embed_dim`** — same in `high_policy` and `low_policy` if you want aligned semantics.
4. **`bottleneck_dim`** = `policy.down_dims[-1]` = `termination_head.bottleneck_dim`.
5. **`global_feat_dim`** — leave null for π_high; verify it matches DP encoder after first load.
6. **`n_action_steps`** — replan window for cached chunks (1 = closed-loop training/eval style).
7. **`option_balance`** / **`termination_pos_weight`** — default yaml enables inverse-frequency skill loss weighting and auto β BCE `pos_weight`; see [`soda/training/README.md`](../soda/training/README.md).

Saved in checkpoints: `high_policy_config`, `low_policy_config`, full `cfg`, normalizers — eval reload uses these.

## See also

- [`soda/training/README.md`](../soda/training/README.md) — train scripts and checkpoint format
- [`soda/models/README.md`](../soda/models/README.md) — module APIs
- [`project_plan.md`](../project_plan.md) §4.3, §8 — locked design choices
