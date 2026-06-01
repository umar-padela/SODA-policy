# receding_horizon_study

Tests how the number of actions executed before replanning (`n_action_steps`) affects
Push-T success rate. Runs on the **best overall SODA policy** (best kernel + best termination
from termination_study). The **best n_action_steps is used only for comparison_study**.

`n_action_steps` is held constant at 8 throughout kernel_size_study and termination_study
so that those experiments are directly comparable. This study tunes it once at the end.

## Prerequisites

- termination_study must be complete (best checkpoint identified)

## Commands

### Step 1: Sweep n_action_steps on best termination_study checkpoint
```bash
# Run evals (incremental — safe to re-run)
modal run --detach \
  experiments/final_experiments/pusht/receding_horizon_study/modal_eval_n_action_steps.py \
  --low-checkpoint /experiments/final_experiments/pusht/termination_study/<stage>/<run>/best.ckpt \
  --config configs/pusht/<best_config>.yaml

# Download latest results + worst-5 videos + regenerate plots without triggering new evals
modal run \
  experiments/final_experiments/pusht/receding_horizon_study/modal_eval_n_action_steps.py \
  --download-only
```

### Step 2: Plot and choose best n_action_steps (local only, no Modal needed)
```bash
python experiments/final_experiments/pusht/receding_horizon_study/plot_n_action_steps.py \
  --data experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/sweep_results.json
```

## Outputs

**Modal Volume** and **local mirror** use the same folder name (`receding_horizon_study/`) — no path mapping needed.

```
receding_horizon_study/
  debug_videos/                  ← worst-5 failure videos per n_action_steps label; < 20% overlap only
    n2/
    n4/
    n6/
    n8/
    n10/
    n16/
    open_loop/
  n_action_steps_sweep/
    sweep_results.json           ← authoritative results (volume) / local mirror
    debug_videos/                ← local download of worst-5 videos
    plot_n_action_steps.png      ← generated locally
```

## Decision

Pick the n_action_steps with the highest mean_score. If scores are within std, prefer
a smaller value (fewer steps = faster to detect suboptimal options = tighter control).

**Pass the chosen value** as `--n-action-steps` to `comparison_study/modal_eval_final.py`.
