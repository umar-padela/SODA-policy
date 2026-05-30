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
modal run \
  experiments/final_experiments/pusht/receding_horizon_study/modal_eval_n_action_steps.py \
  --low-checkpoint /experiments/final_experiments/pusht/termination_study/<stage>/<run>/best.ckpt \
  --config configs/pusht/<best_config>.yaml
```

### Step 2: Plot and choose best n_action_steps
```bash
python experiments/final_experiments/pusht/receding_horizon_study/plot_n_action_steps.py \
  --data experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/sweep_results.json
```

## Outputs

**Modal Volume:**
```
/experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/
  n2/summary.json      ← 50-episode eval results per n_action_steps value
  n4/summary.json
  n6/summary.json
  n8/summary.json
  n10/summary.json
  n16/summary.json
  open_loop/summary.json
```

**Local:**
```
experiments/final_experiments/pusht/receding_horizon_study/
  n_action_steps_sweep/
    sweep_results.json   ← aggregated results for plotting
    plot_n_action_steps.png
```

## Decision

Pick the n_action_steps with the highest mean_score. If scores are within std, prefer
a smaller value (fewer steps = faster to detect suboptimal options = tighter control).

**Pass the chosen value** as `--n-action-steps` to `comparison_study/modal_eval_final.py`.
