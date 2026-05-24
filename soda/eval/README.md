# Push-T eval (`soda/eval`)

Push-T simulation evaluation: load checkpoints, run rollouts in Columbia's env stack, compute overlap metrics, write logs and MP4s.

**P0 protocol (locked):** **receding-horizon control** — predict a full action chunk, execute **`n_action_steps=8`**, replan (same for frozen DP and SODA). β is still checked every sim step for SODA; diffusion replans at most every 8 steps unless β fires earlier.

## Layout

```text
run_eval.py          orchestrator (CLI + EvalConfig)
policy_loaders.py    checkpoint → policy objects
dp_runner.py         frozen DP eval (PushTImageRunner)
soda_runner.py       SODA hierarchical eval (same env stack)
low_only_runner.py   π_low with fixed ω (random test seeds)
segment_rollout.py   expert-anchored segment rollout + side-by-side video
segment_zarr.py      zarr segment index + expert frame/state helpers
metrics.py           overlap @ 125–300 (25-step grid); mean ± std over episodes
run_naming.py        descriptive output folder names
runner_common.py     seeds, action slicing, log serialization
pusht_image_runner.py  DP runner + AsyncVectorEnv shim
gym_vector_compat.py   gym / vector-env patches for Modal
pusht_rollout.py     legacy single-env loop (prefer dp_runner)
dp_frozen.py       frozen Columbia checkpoint URL + download helpers
```

## Entry points

**Local:**

```bash
python soda/eval/run_eval.py --config configs/pusht/dp_frozen.yaml --output-dir /tmp/eval_out
```

**Expert-anchored π_low segment rollout** (zarr segment start state; side-by-side MP4):

```bash
python scripts/rollout_segment.py --list-segments --episode 100 --option-id 0
python scripts/rollout_segment.py --config configs/pusht/soda_supervised.yaml \
  --checkpoint experiments/train_low/soda_supervised/best.ckpt \
  --episode 100 --option-id 0
```

**Modal:**

```bash
modal run modal/modal_eval.py --config configs/pusht/dp_frozen.yaml
modal run modal/modal_eval.py --config configs/pusht/soda_supervised.yaml
modal run modal/modal_eval.py --config configs/pusht/dp.yaml --checkpoint /experiments/train_dp/dp/latest.ckpt
```

## Runners

### `dp_runner.py`

- Loads DP via `load_dp_image_policy_and_cfg`
- Loop: `predict_action` → slice first `n_action_steps` actions → `env.step`
- `n_action_steps` comes from yaml top-level key (**eval only**; default 8)

### `soda_runner.py`

- Loads `HierarchicalPolicy` via `load_soda_policy_and_cfg`
- Same receding-horizon window: `HierarchicalPolicyConfig.n_action_steps` from eval yaml (default 8)
- β checked every sim step; diffusion replans when cache exhausted, β fires, or after `n_action_steps`

### `segment_rollout.py` + `scripts/rollout_segment.py`

- Pick an option segment from zarr (`--segment-index` or `--episode` + `--option-id`)
- Reset Push-T sim via `PushTEnv._set_state` to the expert state at segment start
- Roll π_low with fixed ω; default length = remaining expert segment frames
- Write side-by-side **expert (zarr) | policy (sim)** MP4 under `experiments/segment_rollout/`

## Config param scope

See `configs/pusht/*.yaml` — each line is tagged `[train]`, `[eval]`, or `[train+eval]`.

| Key | Scope | Meaning |
|-----|-------|---------|
| `n_action_steps` | **eval** | Execute window before replan (h=8) |
| `horizon` | train+eval | Diffusion chunk length (16 for DP; auto for SODA) |
| `task.env_runner.*` | eval | Sim rollout / video settings |
| `task.dataset.*` | train | Zarr + dataloader |
| `train_*` / `train_dp` | train | Optimization loop |

## Submodule requirement

Eval requires `third_party/diffusion_policy` (Columbia DP fork). Initialize with:

```bash
git submodule update --init third_party/diffusion_policy
```

## Metrics and artifacts

Each eval run writes under `experiments/eval/{config_stem}/{YYYYMMDD}/{HHMMSS}/`:

| File | Contents |
|------|----------|
| `eval_log.json` | Full run record: config, aggregate `mean_score` / `std_score`, `mean_score@{125,150,…,300}`, and `metrics.episodes` |
| `episodes.json` | Per-episode overlap only (seed, `max_overlap_full`, `max_overlap_at_step`, `success`) |
| `*_runner_log` | Raw DP/SODA runner keys, e.g. `test/sim_max_reward_{seed}` (0–1 max coverage) |
| `media/*.mp4` | Rollout videos renamed to `*_ep{idx}_seed{seed}_score{pct}.mp4` |

Aggregate scores use **sample std** (ddof=1) across test episodes, on the same 0–1 scale as `mean_score`. W&B logs `std_score` and `std_score@{t}` alongside the means.
