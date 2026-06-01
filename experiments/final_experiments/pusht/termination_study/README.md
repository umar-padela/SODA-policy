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

### Stage 1a + 1b: Eval together (recommended)

```powershell
# Run all evals (incremental — skips already-done checkpoints)
modal run --detach `
  experiments/final_experiments/pusht/termination_study/modal_eval_stage1_all.py

# Download latest results + worst-5 videos + regenerate plots without triggering new evals
modal run `
  experiments/final_experiments/pusht/termination_study/modal_eval_stage1_all.py --download-only
```

Results JSON on volume: `final_experiments/pusht/termination_study/stage1/stage1_results.json`
and `stage1b/stage1b_results.json`. Local mirror: same path under `experiments/`.

### Stage 1a + 1b: Eval individually (if needed)

```powershell
modal run --detach `
  experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/modal_eval_stage1.py
modal run --detach `
  experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/modal_eval_stage1b.py
```

### Stage 1a + 1b: Regenerate plots from local data (no Modal)

```powershell
python experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/plot_stage1.py `
  --data experiments/final_experiments/pusht/termination_study/stage1/stage1_results.json `
  --output-dir experiments/final_experiments/pusht/termination_study/stage1

python experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/plot_stage1b.py `
  --data experiments/final_experiments/pusht/termination_study/stage1b/stage1b_results.json `
  --output-dir experiments/final_experiments/pusht/termination_study/stage1b
```

Plots land in `stage1/` and `stage1b/` alongside the JSONs. Each stage produces:
- `plot_stage1_score_vs_epoch.png` — mean ± std vs epoch; gray dashed line = duration termination baseline (97.97%)
- `plot_stage1_best.png` — bar chart of best epoch per run

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

## Path mapping: Modal volume vs local repo

The Modal volume uses short stage names; the local repo uses descriptive folder names.
Scripts handle the translation automatically — do not mix them up manually.

| Modal volume path | Local repo path |
|---|---|
| `stage1/` | `stage1_bottleneck_source/` |
| `stage1b/` | `stage1b_obs_signal/` |
| `stage2/` | `stage2_input_comparison/` |
| `stage3/` | `stage3_joint/` |

**Within each stage on the volume:**
```
stage1/
  bottleneck_expert/       ← training checkpoints (written by train scripts)
  bottleneck_ddim_positive/
  debug_videos/            ← worst-5 failure videos; episodes < 20% overlap only
  stage1_results.json      ← eval results (authoritative)
```

**Locally (same structure, different folder name):**
```
stage1_bottleneck_source/
  debug_videos/            ← downloaded from Modal by eval scripts
  stage1_results.json      ← downloaded from Modal by eval scripts
  plot_stage1_*.png        ← generated locally from the JSON
```

### Copying files within the Modal volume (no `cp` command — download then re-upload)

```powershell
! modal volume get soda-experiments final_experiments/pusht/termination_study/stage1/stage1_results.json tmp.json
! modal volume put soda-experiments tmp.json final_experiments/pusht/termination_study/stage1/stage1_results.json
Remove-Item tmp.json
```

## Beta termination notes

- `beta_transition=0.92` set in `inference.beta_transition` in each `exp_term_*.yaml` config
- After any option switch, the post-replan β check is skipped for one step (`_fresh_option` flag in `LowLevelChunkExecutor`) — guarantees at least one action is taken before β can fire again
- Videos saved only for episodes below 20% overlap (`video_failure_threshold=20.0` default in `rollout_hierarchical`)
