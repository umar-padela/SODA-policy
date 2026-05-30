# termination_study

Four-stage ablation of the β (termination) head. 12 training runs × 100 epochs = 1200 epochs total.
All stages: frozen backbone (stages 1a/1b/2) or joint stop_grad=False (stage 3).
All evals: **β termination** (`duration_termination=False`), `n_action_steps=8`, 50 episodes.

`pos_weight: null` in all configs → auto-calculated from data at training time:
  - Completion-only: `(L-1)/1` ≈ 16 for avg segment length L=17
  - Escape expand-all (K=3): `(L-1)/((L-1)(K-1)+K)` = 16/35 ≈ 0.46

## Stage overview

**Stage 1a (Q1)** — Does DDIM bottleneck beat expert? (2 new runs)
  - `bottleneck_expert`: expert actions at t=0, completion-only
  - `bottleneck_ddim_positive`: DDIM-5 generated actions, completion-only (closes train/inference gap)
  → Winner = `bottleneck_best`

**Stage 1b (Q2)** — Does positive_negative beat positive? (3 new + 1 reuse)
  - `bottleneck_ddim_positive` — **reused from 1a**, no retraining
  - `bottleneck_ddim_positive_negative` — DDIM-5, escape relabeling (expand-all)
  - `obs_positive` — obs-based β, completion-only
  - `obs_positive_negative` — obs-based β, escape relabeling (expand-all)
  → Separate winners for bottleneck and obs sides

**Stage 2 (Q3)** — Is bottleneck or obs better? Does combining help? (1 new run)
  - `bottleneck_best` — reused from 1a/1b
  - `obs_best` — reused from 1b
  - `both` — concat(bottleneck_best, obs_best): 1 new training run
  → Compare all three + duration_termination baseline from kernel_size_study

**Stage 3 (Q1–Q3 repeated, stop_grad=False, optional)** — Does joint training improve? (6 new runs)
  - From 1a: `bottleneck_expert_joint`, `bottleneck_ddim_positive_joint`
  - From 1b: `bottleneck_ddim_positive_negative_joint`, `obs_positive_joint`, `obs_positive_negative_joint`
  - From 2:  `both_joint`
  → λ=0.01 for all joint runs

## Prerequisites

- kernel_size_study complete; best kernel and checkpoint known
- Update `policy.kernel_size` in all `configs/pusht/exp_term_*.yaml` before running

---

## Commands

### Stage 1a + 1b: Train in parallel (5 runs simultaneously)

```powershell
# Stage 1a: Q1 — bottleneck source (2 runs)
modal run --detach `
  experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/modal_train_stage1.py `
  --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt `
  --run-readme "termination_study stage1a bottleneck source"

# Stage 1b: Q2 — escape signal comparison (3 new runs; runs simultaneously with 1a)
modal run --detach `
  experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/modal_train_stage1b.py `
  --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt `
  --run-readme "termination_study stage1b escape signal"
```

### Stage 1a: Eval + pick bottleneck_best

```powershell
modal run `
  experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/modal_eval_stage1.py

python experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/plot_stage1.py `
  --data experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/stage1_results.json
```

### Stage 1b: Eval + pick obs_best (and whether escape helps bottleneck)

```powershell
modal run `
  experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/modal_eval_stage1b.py

python experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/plot_stage1b.py `
  --data experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/stage1b_results.json
```

---

### Stage 2: Train (1 new run — `both` combination)

```powershell
modal run --detach `
  experiments/final_experiments/pusht/termination_study/stage2_input_comparison/modal_train_stage2.py `
  --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt `
  --bottleneck-winner-config exp_term_bottleneck_{expert|ddim_positive|ddim_positive_negative} `
  --obs-escape {True|False} `
  --run-readme "termination_study stage2 both combination"
```

### Stage 2: Eval

```powershell
modal run `
  experiments/final_experiments/pusht/termination_study/stage2_input_comparison/modal_eval_stage2.py `
  --bottleneck-best-dir /experiments/final_experiments/pusht/termination_study/stage1{a|b}/{winner}/ `
  --obs-best-dir /experiments/final_experiments/pusht/termination_study/stage1b/{obs_winner}/ `
  --duration-baseline-ckpt /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt
```

---

### Stage 3: Train (5 runs in parallel, then 1 after decisions)

```powershell
# Step 1: repeat 1a+1b with joint training (5 runs in parallel)
modal run --detach `
  experiments/final_experiments/pusht/termination_study/stage3_joint/modal_train_stage3.py `
  --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt `
  --group stage1 `
  --run-readme "termination_study stage3 joint stage1 runs"

# Step 2: both_joint after 1a/1b decisions
modal run --detach `
  experiments/final_experiments/pusht/termination_study/stage3_joint/modal_train_stage3.py `
  --base-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k{N}/best.ckpt `
  --group stage2 --obs-escape {True|False} `
  --run-readme "termination_study stage3 joint both"
```

### Stage 3: Eval

```powershell
modal run `
  experiments/final_experiments/pusht/termination_study/stage3_joint/modal_eval_stage3.py
```

---

## Decision

Best policy = highest mean_score across all stages. Pass to `receding_horizon_study` then `comparison_study`.
If no β variant beats duration_termination baseline by >1 std → use duration termination (valid negative result).

## Volume layout

```
/experiments/final_experiments/pusht/termination_study/
  stage1/
    bottleneck_expert/
    bottleneck_ddim_positive/
  stage1b/
    bottleneck_ddim_positive_negative/
    obs_positive/
    obs_positive_negative/
  stage2/
    both/
  stage3/
    bottleneck_expert_joint/
    bottleneck_ddim_positive_joint/
    bottleneck_ddim_positive_negative_joint/
    obs_positive_joint/
    obs_positive_negative_joint/
    both_joint/
```
