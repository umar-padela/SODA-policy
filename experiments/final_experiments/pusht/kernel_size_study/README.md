# kernel_size_study

Tests whether the U-Net temporal convolution kernel size (5, 7, or 9) affects
policy quality on Push-T. All three kernels are trained from scratch and evaluated
with an epoch sweep (every 50 epochs) using duration termination (no β) and
`n_action_steps=8` fixed. The **best kernel size is used for all termination_study runs**.

`n_action_steps` is held at 8 throughout kernel_size_study and termination_study
so that the termination head comparison is apples-to-apples with no-termination runs.
It is tuned in receding_horizon_study only after the best overall policy is identified.

## Prerequisites

None. This is the first experiment.

## Commands

### Step 1: Train k=5, k=7, k=9 (500 epochs each, all in parallel, ~3–4 hrs)
```bash
modal run --detach \
  experiments/final_experiments/pusht/kernel_size_study/modal_train_kernels.py \
  --run-readme "kernel_size_study k=5 k=7 k=9 no-beta 500ep"
```

### Step 2: Epoch sweep eval for all three kernels (n_action_steps=8 fixed)
```bash
# Run evals (incremental — safe to re-run)
modal run --detach \
  experiments/final_experiments/pusht/kernel_size_study/modal_eval_kernel_sweep.py

# Download latest results + worst-5 videos + regenerate plots without triggering new evals
modal run \
  experiments/final_experiments/pusht/kernel_size_study/modal_eval_kernel_sweep.py \
  --download-only
```

### Step 3: Plot and choose best kernel (local only, no Modal needed)
```bash
python experiments/final_experiments/pusht/kernel_size_study/plot_kernel_study.py \
  --data experiments/final_experiments/pusht/kernel_size_study/kernel_sweep_results.json
```

## Outputs

**Modal Volume** and **local mirror** use the same folder name (`kernel_size_study/`) — no path mapping needed.

```
kernel_size_study/
  k5/                            ← k=5 checkpoints (volume only)
  k7/
  k9/
  debug_videos/                  ← worst-5 failure videos per (kernel, epoch); < 20% overlap only
    k5/epoch_0050/
    k7/epoch_0050/
    ...
  kernel_sweep_results.json      ← authoritative results (volume) / local mirror
  plot_kernel_vs_epoch_*.png     ← generated locally from the JSON
  plot_kernel_best.png
```

## Decision

Choose the kernel with the highest best-epoch mean_score.
**Update `policy.kernel_size` in all termination study configs** before running stage 1:
- `configs/pusht/exp_term_bottleneck_expert.yaml`
- `configs/pusht/exp_term_bottleneck_ddim_positive.yaml`
- `configs/pusht/exp_term_obs.yaml`
- `configs/pusht/exp_term_both.yaml`
- `configs/pusht/exp_term_obs_joint.yaml`
- `configs/pusht/exp_term_both_joint.yaml`

Also update the `--base-checkpoint` path in termination_study commands to point to
`/experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt`.
