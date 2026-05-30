# `soda/inference`

Runtime policy logic: how trained π_high and π_low are composed at **decision time**. Does not run the sim loop (see [`eval/`](../eval/README.md)) and does not define network weights (see [`models/`](../models/README.md)).

## Files

| File | Purpose |
|------|---------|
| [`low_level_executor.py`](low_level_executor.py) | `LowLevelChunkExecutor` — shared π_low cache, β, native chunk execute |
| [`fixed_option_low_controller.py`](fixed_option_low_controller.py) | Fixed-ω wrapper for segment rollouts |
| [`hierarchical_controller.py`](hierarchical_controller.py) | `HierarchicalPolicy` — π_high + executor |

## `HierarchicalPolicy`

Drop-in replacement for DP `BaseImagePolicy.predict_action(obs_dict)` so `soda_runner` can reuse the Columbia env stack.

### Attached models

- `pi_high` — `HighPolicy.sample_option(encode_obs(obs))`
- `pi_low` — diffusion + `predict_beta`

Constructed by [`load_soda_policy_and_cfg`](../eval/policy_loaders.py) after loading both checkpoints.

### Per sim step

Called **once per env step** (runner executes a single action per call):

1. **β** — every step (including immediately after replan): `pi_low.predict_beta(obs, ω, cached_action_pred, t=0)`
2. **Resample ω** if `β > beta_transition` (segment exit — only way to leave current low-level segment)
3. **Replan diffusion** if cache is empty or the decompressed native chunk is exhausted (same ω)
4. **Return** next row from `action_unstretched`

Low-level segments are **not** capped at `n_action_steps` when a native chunk is available — the full decompressed chunk runs out before replanning. If the chunk is shorter than 8 steps and β has not fired, π_low replans under the same ω.

### Parameters

| Field | Scope | Default | Meaning |
|-------|-------|---------|---------|
| `n_action_steps` | **eval** | 8 | Legacy DP yaml field; native SODA execute uses full decoded chunk length |
| `beta_transition` | eval | 0.5 | Segment ends when β exceeds this |
| `beta_diffusion_t` | eval | 0 | Diffusion timestep for cheap β forward |
| `env_action_dim` | eval | 2 | Push-T xy (drop duration channel) |

β is checked **every sim step** while executing a cached chunk (including right after replan). Diffusion replans when the native chunk is exhausted; π_high resamples ω only when β fires.

## `reset()`

Clears option, cached chunk, cursor; calls `reset()` on sub-policies if present. Called at episode start by the runner.

## Tests

```bash
pytest tests/test_hierarchical_controller.py
```

Uses mock π_high / π_low (no GPU, no DP submodule).

## See also

- [`eval/policy_loaders.py`](../eval/policy_loaders.py) — load weights into this controller
- [`models/low_policy.py`](../models/low_policy.py) — `predict_beta` / `predict_action` implementation
