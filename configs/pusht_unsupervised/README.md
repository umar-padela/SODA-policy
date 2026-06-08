# Push-T Unsupervised Study (E3)

Training and evaluation runbook for **experiment E3**: Push-T with LOVE unsupervised option discovery.

Two config files live here:

| File | Purpose |
|---|---|
| `unsupervised_k5_no_beta.yaml` | π_low training — k=5 U-Net, LOVE labels, 500 epochs, cosine LR, duration termination |
| `unsupervised_high_starts_prev_opt.yaml` | π_high training — segment starts + prev-option conditioning, LOVE labels, 1000 epochs |

Both are identical to their supervised counterparts (`configs/pusht/exp_k5_no_beta.yaml` and
`configs/pusht/high_starts_prev_opt.yaml`) except `task.dataset.option_id_key` is set to
`option_id_unsupervised`, which reads LOVE-generated segment labels from the zarr.

**Prerequisite:** LOVE labels must be written into `data/raw/pusht/pusht.zarr` under the
`option_id_unsupervised` column before training. See `soda/option_discovery/unsupervised/` and
the LOVE adapter for label generation.

**No ordering constraint:** π_low and π_high can be trained in parallel. π_high uses a fresh
ResNet initialized from ImageNet weights (`imagenet_init: true`) — it does not depend on the
π_low checkpoint for weight initialization.

---

## 1. Train π_low (LOVE labels, k=5, 500 epochs)

```bash
modal run --detach modal/modal_train_low.py \
  --config-name unsupervised_k5_no_beta \
  --task pusht_unsupervised \
  --hydra-overrides "train_low.output_dir=/experiments/final_experiments/pusht/unsupervised_study/low train_low.wandb_run_name=unsupervised_k5_no_beta_500ep" \
  --run-readme "E3 LOVE unsupervised k5 no-beta 500ep"
```

Checkpoints saved under:

```
/experiments/final_experiments/pusht/unsupervised_study/low/
  best.ckpt
  epoch_0050.ckpt
  epoch_0100.ckpt
  ...
  epoch_0500.ckpt
```

W&B: project `soda-final-experiments`, group `unsupervised_study`, tags `[unsupervised_study, love, k5]`.

---

## 2. Train π_high (LOVE labels, segment starts + prev-option)

Can be launched in parallel with π_low — no ordering dependency.

```bash
modal run --detach modal/modal_train_high.py \
  --config-name unsupervised_high_starts_prev_opt \
  --task pusht_unsupervised
```

Checkpoints saved under:

```
/experiments/final_experiments/pusht/unsupervised_study/high/
  best.ckpt
  epoch_0005.ckpt
  epoch_0010.ckpt
  ...
```

W&B: project `soda-train-high`, group `unsupervised_study`, tags `[unsupervised_study, love, high]`.

---

## 3. Evaluation

Evaluation runs the hierarchical controller (π_high + π_low) for 50 episodes, 300 max steps,
seed=100000 — matching the locked P0/E1 eval protocol.

Key eval parameters:

```python
N_EPISODES           = 50
TEST_START_SEED      = 100000
MAX_STEPS            = 300
N_ACTION_STEPS       = 8         # receding horizon
DURATION_TERMINATION = True      # chunk-duration signal; no learned β
HIGH_CHECKPOINT      = "/experiments/final_experiments/pusht/unsupervised_study/high/best.ckpt"
```

### Epoch sweep (low policy)

Write a dedicated eval script in `experiments/final_experiments/pusht/unsupervised_study/`
following the pattern in `experiments/final_experiments/pusht/kernel_size_study/modal_eval_kernel_sweep.py`:
1. Discover all `epoch_NNNN.ckpt` files in `unsupervised_study/low/`.
2. Skip already-evaluated checkpoints (incremental, crash-safe).
3. Spawn `rollout_hierarchical` calls in parallel on Modal.
4. Save results JSON and download locally.

### Smoke test (5 episodes)

```bash
modal run modal/modal_eval.py \
  --checkpoint /experiments/final_experiments/pusht/unsupervised_study/low/best.ckpt \
  --high-checkpoint /experiments/final_experiments/pusht/unsupervised_study/high/best.ckpt \
  --n-episodes 5
```

---

## 4. Plotting and Comparison

### Epoch-vs-score curves

```bash
python experiments/final_experiments/pusht/unsupervised_study/plot_unsupervised_study.py \
  --data experiments/final_experiments/pusht/unsupervised_study/unsupervised_results.json
```

### Supervised vs. unsupervised comparison

To overlay E3 (LOVE) against E1 (supervised k=5) in a single figure:

```python
runs = {
    "supervised E1 k=5": supervised_k5_results,
    "unsupervised E3 LOVE k=5": unsupervised_results,
}
plot_eval_curve(runs, output_path, ylabel="Mean Max Overlap (%, t≤300, 50ep)", ylim=(85, 100))
```

Style guide: `experiments/final_experiments/pusht/PLOT_STYLE_GUIDE.md`.

---

## 5. Differences from supervised configs

| Field | Supervised (E1) | Unsupervised (E3) |
|---|---|---|
| `task.dataset.option_id_key` | `option_id_supervised` | `option_id_unsupervised` |
| Low config file | `configs/pusht/exp_k5_no_beta.yaml` | `configs/pusht_unsupervised/unsupervised_k5_no_beta.yaml` |
| High config file | `configs/pusht/high_starts_prev_opt.yaml` | `configs/pusht_unsupervised/unsupervised_high_starts_prev_opt.yaml` |
| High checkpoint used at eval | `high_study/high_starts_prev_opt/best.ckpt` | `unsupervised_study/high/best.ckpt` |
| `train_low.wandb_group` | `kernel_size_study` | `unsupervised_study` |
| `train_low.wandb_tags` | `[kernel_size_study, k5]` | `[unsupervised_study, love, k5]` |

All other hyperparameters (kernel size, horizon, LR schedule, eval protocol) are identical.

---

## 6. Troubleshooting

**`KeyError: option_id_unsupervised`** — LOVE labels have not been written to the zarr yet.
Run the LOVE adapter first: `soda/option_discovery/unsupervised/love_adapter/`.

**`num_options` mismatch** — `num_options: null` means it is inferred from the zarr at load time.
LOVE may produce a different number of options than the supervised heuristic (3). π_high must
be trained on LOVE labels (this config) rather than reusing the supervised `high_starts_prev_opt`
checkpoint — the option embedding dimensions will not match.

**Eval scores lower than E1** — Expected if LOVE segments are noisier than heuristic labels.
Report the number of LOVE options and mean segment length alongside the score.
