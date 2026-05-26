# LOVE adapter (E3 — unsupervised option discovery)

Trains [LOVE](../../../../third_party/love) (`hssm_rl.EnvModel`) on Push-T
demonstrations and writes `data/option_id_unsupervised` into
`data/raw/pusht/pusht.zarr`. Consumed by `configs/pusht/soda_unsupervised.yaml`.

## Quick start

```bash
# 1. train (on Modal, T4 GPU, ~hours)
modal run modal/modal_train_love.py

# 2. label (local; reads ckpt from the Modal volume or from experiments/love_pusht/)
python -m soda.option_discovery.unsupervised.love_adapter.label \
    --ckpt experiments/love_pusht/best.ckpt

# 3. verify
python -c "import zarr; r = zarr.open('data/raw/pusht/pusht.zarr'); \
    print(list(r['data'].keys())); print(r['data']['option_id_unsupervised'][:50])"

# 4. update configs/pusht/soda_unsupervised.yaml: num_options = K_final
#    (label.py prints K_final at the end)
```

## What it does

- Wraps `pusht.zarr` as fixed-length state/action windows (`dataset.py`).
- **Discretizes 2-D continuous Push-T actions** into a `num_action_bins`
  codebook with numpy k-means (`quantize.py`). This is required because
  `hssm_rl.py:803` reconstructs actions with `F.cross_entropy` and only
  accepts integer ids — i.e. LOVE upstream is architecturally
  discrete-action-only. Centroids are persisted alongside the checkpoint so
  labeling uses the same codebook.
- Imports `EnvModel` from `third_party/love/hssm_rl.py` (matching upstream
  `train_rl.py`, NOT `hssm_v2.py`) and reuses upstream
  `GridActionEncoder` + `GridDecoder`. Only the state encoder is ours
  (`encoders.StateEncoder` — small MLP for the 5-dim state).
- Trains with the LOVE objective via `out["train_loss"]`. The model
  precomputes the full objective internally; we don't reimplement it.
- At inference, tiles non-overlapping windows (`stride=seq_size`) over each
  full episode, takes `out["option_list"]` per window (covers the inner
  `seq_size` frames; the `init_size` edges at episode ends are
  forward/back-filled from the nearest labeled frame).
- Filters options with empirical marginal < `min_marginal` (default 0.5%)
  and compacts ids to `0..K_final-1` before writing back into the zarr.

## Tuning knobs (`config.py: LoveConfig`)

| Field                          | Default | Notes                                       |
|--------------------------------|---------|---------------------------------------------|
| `latent_n`                     | 10      | Upper bound on # of options                 |
| `num_action_bins`              | 16      | Action codebook size (k-means centroids)    |
| `coding_len_coeff`             | 0.005   | Description-length weight                   |
| `kl_coeff`                     | 0.0     | LOVE recommends 0 when coding-len > 0       |
| `rec_coeff`                    | 1.0     | Action-reconstruction weight                |
| `use_min_length_boundary_mask` | True    | Prevents 1-step degenerate skills           |
| `seq_size` / `init_size`       | 6 / 1   | Inner / burn-in lengths (window_len = 8)    |
| `seg_len` / `seg_num`          | 100     | Per-episode segment ceilings                |
| `batch_size` / `learn_rate`    | 64 / 5e-4 | Optimizer (Adam + amsgrad)                |
| `min_marginal`                 | 0.005   | Drop rarely-used options                    |

**If LOVE collapses to 1 option:** raise `coding_len_coeff` to 0.01 or
`latent_n` to 16.

**If LOVE uses every option roughly uniformly with no temporal structure:**
verify `use_min_length_boundary_mask=True`. Raise `coding_len_coeff` to 0.02.

**If 1-step skills appear:** verify `use_min_length_boundary_mask=True`;
recheck `rec_coeff`/`kl_coeff` balance against the upstream grid_world recipe
in `third_party/love/README.md`.

Target post-filter `K_final` is **3–10** (Umar's heuristic produces 4 —
STATIONARY, REPOSITION, LINEAR-PUSH, PIVOT-PUSH — so a similar ballpark
gives a fair comparison).

## Caveats

- **State-only input, action codebook on the side.** The original LOVE was
  developed for low-dim discrete-action domains (grid_world). Push-T's
  continuous 2-D actions force us to discretize. We picked state-only obs
  per `project_plan.md` §8F — an image encoder adapter is left as future
  work and would replace `encoders.StateEncoder`.
- **`hssm_rl.py`, not `hssm_v2.py`.** The project plan references `hssm_v2.py`
  but upstream `train_rl.py` actually uses `hssm_rl.py` and the two have
  different `EnvModel.__init__` signatures (`hssm_v2` lacks
  `latent_n`/`rec_coeff`/`kl_coeff`/`use_min_length_boundary_mask`). We
  follow upstream.
- **LOVE upstream is pinned to Torch 1.10–1.12 era.** It's invoked via
  `sys.path.insert(0, third_party/love)` rather than packaged as a module.
  The Modal image (`environment.modal.yml`, Torch 1.12.1) is the supported
  training environment; local CPU runs work for smoke tests only.
