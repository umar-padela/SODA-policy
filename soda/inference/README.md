# `soda/inference`

Runtime policy logic: how trained π_high and π_low are composed at **decision time**. Does not run the sim loop (see [`eval/`](../eval/README.md)) and does not define network weights (see [`models/`](../models/README.md)).

## Files

| File | Purpose |
|------|---------|
| [`hierarchical_controller.py`](hierarchical_controller.py) | `HierarchicalPolicy` — option loop + β + chunk cache |

## `HierarchicalPolicy`

Drop-in replacement for DP `BaseImagePolicy.predict_action(obs_dict)` so `soda_runner` can reuse the Columbia env stack.

### Attached models

- `pi_high` — `HighPolicy.sample_option(encode_obs(obs))`
- `pi_low` — diffusion + `predict_beta`

Constructed by [`load_soda_policy_and_cfg`](../eval/policy_loaders.py) after loading both checkpoints.

### Per sim step

Called **once per env step** (runner executes a single action per call):

1. **β** — if a cached chunk exists: `pi_low.predict_beta(obs, ω, cached_action_pred, t=0)` → termination head
2. **Resample ω** if `β > beta_transition`
3. **Replan diffusion** if no cache, β fired, or `cursor >= n_action_steps` (eval yaml)
4. **On replan:** `pi_low.predict_action` → cache full horizon chunk + `action_pred`
5. **Return** next row from cached chunk (duration channel stripped for env)

Training predicts a **full action chunk** (variable horizon for SODA). The execute window before replanning is **`HierarchicalPolicyConfig.n_action_steps`** from eval yaml (**default 8**, receding-horizon — same protocol as vanilla DP).

### Parameters

| Field | Scope | Default | Meaning |
|-------|-------|---------|---------|
| `n_action_steps` | **eval** | 8 | Execute window before diffusion replan (h=8) |
| `beta_transition` | eval | 0.5 | Segment ends when β exceeds this |
| `beta_diffusion_t` | eval | 0 | Diffusion timestep for cheap β forward |
| `env_action_dim` | eval | 2 | Push-T xy (drop duration channel) |

β is checked **every sim step** while executing a cached chunk; diffusion replans at most every `n_action_steps` steps unless β fires earlier.

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
