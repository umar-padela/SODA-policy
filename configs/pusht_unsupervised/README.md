# Push-T Unsupervised Study (E3)

Training and evaluation runbook for **experiment E3**: Push-T with LOVE unsupervised option discovery.

The single config file here (`unsupervised_k9_no_beta.yaml`) is identical to the best supervised
baseline (`configs/pusht/exp_k9_no_beta.yaml`) except for one key: `task.dataset.option_id_key`
is set to `option_id_unsupervised`, which reads LOVE-generated segment labels from the zarr
instead of the heuristic supervised labels used in E1.

**Prerequisite:** LOVE labels must be written into `data/raw/pusht/pusht.zarr` under the
`option_id_unsupervised` column before training. See `soda/option_discovery/unsupervised/` and
the LOVE adapter for label generation.

---

## 1. Training

### One-liner (Modal GPU)

```bash
# From repo root — train π_low with LOVE option labels
modal run --detach modal/modal_train_low.py \
  --config-name unsupervised_k9_no_beta \
  --task pusht_unsupervised \
  --hydra-overrides "train_low.output_dir=/experiments/final_experiments/pusht/unsupervised_study/k9 train_low.wandb_run_name=unsupervised_k9_no_beta_500ep" \
  --run-readme "E3 LOVE unsupervised k9 no-beta 500ep"
```

`--detach` keeps the job alive if the terminal closes. The local client still streams logs
until you Ctrl-C; the remote job continues independently.

### Output layout

Checkpoints are saved to the Modal Volume under:

```
/experiments/final_experiments/pusht/unsupervised_study/k9/
  best.ckpt          # best validation diffusion loss
  epoch_0050.ckpt
  epoch_0100.ckpt
  ...
  epoch_0500.ckpt
  runs/YYYYMMDD_HHMMSS/   # archived copy + README for each run
```

### W&B

Runs appear under **project** `soda-final-experiments`, **group** `unsupervised_study`,
**tags** `[unsupervised_study, love, k9]`.

---

## 2. Validation (training curves)

W&B tracks `loss_diffusion` (primary, used for `best.ckpt`) and `loss_termination` (zero here
since `termination_loss_weight=0.0`). The best checkpoint is selected on val diffusion loss.

To inspect the training run live:

```
https://wandb.ai/your-org/soda-final-experiments
```

Filter by group `unsupervised_study` to see all E3 runs side-by-side.

---

## 3. Evaluation on Modal

Evaluation runs the hierarchical controller (π_high + π_low) for 50 episodes, 300 max steps,
seed=100000. This matches the locked P0/E1 eval protocol.

### Smoke test (5 episodes, fast)

```bash
modal run modal/modal_eval.py \
  --checkpoint /experiments/final_experiments/pusht/unsupervised_study/k9/best.ckpt \
  --n-episodes 5
```

### Full eval (50 episodes)

Write a dedicated eval script in `experiments/final_experiments/pusht/unsupervised_study/`
following the pattern in `experiments/final_experiments/pusht/kernel_size_study/modal_eval_kernel_sweep.py`.
Key parameters to replicate:

```python
N_EPISODES        = 50
TEST_START_SEED   = 100000
MAX_STEPS         = 300
N_ACTION_STEPS    = 8          # receding horizon, fixed across all final experiments
DURATION_TERMINATION = True    # use chunk-duration signal, not learned β
OPEN_LOOP         = False

HIGH_CHECKPOINT = (
    "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt"
)
```

The eval script should:
1. Discover all `epoch_NNNN.ckpt` files in the unsupervised study dir.
2. Skip already-evaluated (epoch) pairs (incremental, crash-safe).
3. Spawn `rollout_hierarchical` calls in parallel on Modal.
4. Collect `max_overlap_full` and `max_overlap_at_step@{t}` metrics.
5. Save results to a JSON on the Modal Volume and download locally.

### Metric to report

**Primary:** `mean max_overlap_full` at the best epoch (t ≤ 300, 50 episodes).  
**Secondary:** `mean_score@{t}` curves at t ∈ {150, 200, 250, 300} for epoch-vs-score plots.

---

## 4. Plotting

### Epoch-vs-score curves

Follow the pattern in `experiments/final_experiments/pusht/kernel_size_study/plot_kernel_study.py`.
Once your results JSON is populated, run:

```bash
python experiments/final_experiments/pusht/unsupervised_study/plot_unsupervised_study.py \
  --data experiments/final_experiments/pusht/unsupervised_study/unsupervised_results.json
```

Expected outputs (save next to the JSON):
- `plot_unsupervised_vs_epoch_t300.png` — mean max overlap vs epoch (t ≤ 300, 50ep) — **main**
- `plot_unsupervised_vs_epoch_t250.png`
- `plot_unsupervised_vs_epoch_t200.png`
- `plot_unsupervised_vs_epoch_t150.png`
- `plot_unsupervised_vs_epoch_grid.png` — all horizons in a subplot grid

### Style guide

See `experiments/final_experiments/pusht/PLOT_STYLE_GUIDE.md` for the full formatting spec.
Key rules:
- No error bands (SEM or std bands are not used).
- `ylim` matched to study range (typically `(88, 100)` for Push-T).
- `Chunk Duration` label (not `Duration Termination`).
- `term_run_style` encoding for line styles.

### Comparison vs supervised baseline

To compare E3 (LOVE unsupervised) against E1 (supervised, best k9 checkpoint) in a single plot,
load both result JSONs and pass them as separate runs to `plot_eval_curve` from
`experiments/final_experiments/pusht/shared/experiment_utils.py`.

```python
runs = {
    "supervised (E1)": supervised_results,
    "unsupervised LOVE (E3)": unsupervised_results,
}
plot_eval_curve(runs, output_path, ylabel="Mean Max Overlap (%, t≤300, 50ep)", ylim=(88, 100))
```

---

## 5. Differences from supervised config

| Field | Supervised (E1) | Unsupervised (E3) |
|---|---|---|
| `task.dataset.option_id_key` | `option_id_supervised` | `option_id_unsupervised` |
| `train_low.wandb_group` | `kernel_size_study` | `unsupervised_study` |
| `train_low.wandb_tags` | `[kernel_size_study, k9]` | `[unsupervised_study, love, k9]` |
| Config file | `configs/pusht/exp_k9_no_beta.yaml` | `configs/pusht_unsupervised/unsupervised_k9_no_beta.yaml` |

All other hyperparameters (kernel size, horizon, LR schedule, eval protocol) are identical.

---

## 6. Troubleshooting

**`KeyError: option_id_unsupervised`** — LOVE labels have not been written to the zarr yet.
Run the LOVE adapter first: `soda/option_discovery/unsupervised/love_adapter/`.

**`num_options` mismatch** — `num_options: null` means it is inferred from the zarr at load time.
LOVE may produce a different number of options than the supervised heuristic. The π_high checkpoint
must match the `num_options` seen during its own training — re-train π_high on LOVE options if
the count differs from the supervised baseline.

**Eval scores lower than E1** — Expected if LOVE segments are noisier than heuristic labels;
report the number of options and mean segment length alongside the score.
