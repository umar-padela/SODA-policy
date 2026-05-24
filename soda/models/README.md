# `soda/models`

Neural network modules for the hierarchical SODA stack. Training loops live in [`soda/training/`](../training/README.md); runtime orchestration in [`soda/inference/`](../inference/README.md).

## Files

| File | Class | Role |
|------|-------|------|
| [`high_policy.py`](high_policy.py) | `HighPolicy` | π_high(ω \| s) via flow matching on option embeddings |
| [`low_policy.py`](low_policy.py) | `LowPolicy` | π_low(a \| s, ω) — DP hybrid U-Net + option embed + D+1 actions |
| [`termination_head.py`](termination_head.py) | `TerminationHead` | β MLP on stop-grad U-Net bottleneck |
| [`unet_bottleneck.py`](unet_bottleneck.py) | helpers | Mid-layer hook for bottleneck features |

## π_high — `HighPolicy`

**Training:** segment-start obs → flow-matching loss to option embedding (see `compute_loss`).

**Inference:** `encode_obs(obs)` → `sample_option(global_feat)` → discrete ω (nearest embedding after Euler integration).

**Vision:** `ObsEncoder` wraps frozen hybrid ResNet + normalizer from π_low (`from_low_policy_checkpoint`) or Columbia DP (`from_dp_checkpoint`).

Config: Hydra `high_policy:` block → `HighPolicyConfig`. **All fields:** [`configs/README.md`](../../configs/README.md#π_high--high_policy--train_high).

## π_low — `LowPolicy`

Subclasses `DiffusionUnetHybridImagePolicy` from `third_party/diffusion_policy` (subclass only in this repo — do not edit upstream).

**Additions vs vanilla DP:**

- `option_embed(ω)` concatenated to `global_cond`
- Action dim **D+1** (xy + duration channel)
- `TerminationHead` on pooled bottleneck (stop-grad before MLP)

**Key APIs:**

```python
predict_action(obs_dict, option_id)  → {"action", "action_pred"}
predict_beta(obs_dict, option_id, action_plan, diffusion_t=0)  → (B,) probabilities
compute_loss(batch)  → diffusion MSE + weighted termination BCE (option + β pos_weight)
```

**Pre-sliced `action`:** indices `[n_obs_steps-1 : n_obs_steps-1+n_action_steps]` — same convention as DP. `HierarchicalPolicy` walks this slice one step at a time while checking β every step.

U-Net `horizon` comes from root yaml `horizon` (DP convention). `null` → longest option segment in the dataset; `train_low` passes the resolved value to `LowPolicy`.

Config: Hydra `low_policy:` + `policy:` + `noise_scheduler:` → constructor args. **All fields:** [`configs/README.md`](../../configs/README.md#π_low--low_policy--policy--noise_scheduler--train_low).

## Termination — `TerminationHead`

Input: pooled U-Net bottleneck `(B, C)` — **detached** during training so β loss does not update the diffusion backbone.

Output: per-env termination logit; `sigmoid` compared to `beta_transition` at inference.

**Training loss:** weighted BCE with PyTorch `pos_weight` (`low_policy.termination_pos_weight`, auto ≈ **17** on Push-T train split = β=0 frames / β=1 frames). Prevents collapse to always predicting 0. Combined with `train_low.option_balance` for per-skill scaling. See [`configs/README.md`](../../configs/README.md).

## Bottleneck hook

`forward_unet_with_bottleneck(model, sample, t, global_cond)` runs one U-Net forward and returns `(prediction, pooled_bottleneck)`.

Used by:

- `LowPolicy.predict_beta` (inference, every sim step — `t=0`, model or expert plan)
- `LowPolicy.compute_loss` — **two** forwards: random-`t` noisy traj for diffusion; `t=0` clean expert traj for termination BCE

## Dependencies

π_low requires `third_party/diffusion_policy` on `PYTHONPATH`. Local import guard: `LowPolicy` class is built lazily via `_build_low_policy_class()`.

## Tests

```bash
pytest tests/test_high_policy.py tests/test_low_policy_init.py tests/test_beta_inference.py
```

## See also

- [`project_plan.md`](../../project_plan.md) §2, §4.2, §8 — locked design choices (stop-grad β, mean-pool duration)
