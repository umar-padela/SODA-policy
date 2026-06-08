# SODA Final Experiments — Push-T Evaluation Study

**Project**: SODA — Supervised Option Discovery for Dynamic Action chunking  
**Task**: Push-T (image observations, 2D end-effector control)  
**Author**: Umar Padela  
**Date**: 2026-05

---

## 1. Motivation and Research Questions

Hierarchical imitation learning decomposes long-horizon tasks into discrete skills (options), with the hypothesis that skill-conditioned low-level policies are simpler to learn than a monolithic policy over the full task. SODA implements this as:

- **π_high**: a cross-entropy classifier that selects an option ω from observations
- **π_low**: a diffusion U-Net that generates variable-length action chunks conditioned on ω
- **β**: a termination head that fires when the current option is complete

The baseline is Columbia Diffusion Policy (DP), which uses a single monolithic diffusion model over the full task without skill decomposition.

This study addresses the following questions, in dependency order:

**Q1 (kernel_size_study)**: Among U-Net temporal kernel sizes {5, 7, 9}, which yields the best diffusion policy for Push-T option segments? — **CONCLUDED**: k=9 (epoch_0450, mean=97.97%, std=12.99%) wins. Used as backbone for all downstream experiments.

**Q2a (termination_study/stage1a)**: Does using policy-generated action plans (DDIM-5) for the bottleneck outperform using expert actions? (Train-inference distribution gap) — **CONCLUDED**: Expert actions win (88.96% vs 84.11%). DDIM bottleneck does not help at this stage.

**Q2b (termination_study/stage1b)**: Does training obs-based β with escape relabeling (wrong-option examples → β=1) improve over boundary-only training? (Completion vs completion+escape) — **CONCLUDED**: Boundary-only (`obs_positive`) wins (85.99% vs 74.92%). Escape relabeling hurts on Push-T.

**Q3 (termination_study/stage2)**: Does combining the best bottleneck source (from Q2a) with the best obs training signal (from Q2b) via feature concatenation outperform either alone? — **SKIPPED**: Duration termination (97.97%) established as winner before Stage 2 could run; not pursued.

**Q4 (termination_study/stage3)**: Does joint training (stop_grad=False, λ=0.01) improve over frozen for each of the above configurations? — **CONCLUDED**: Joint training helps `bottleneck_ddim_positive` (84.11%→89.65%) but hurts `bottleneck_expert` (88.96%→87.32%). Best joint result (89.65%) still trails duration termination by 8+ points.

**Q5 (receding_horizon_study)**: For the best overall SODA policy, what receding-horizon window size (n_action_steps) maximizes Push-T performance? — **CONCLUDED**: n=8 is optimal (mean=97.97%, std=13.00%).

**Q6 (comparison_study)**: Does the best SODA policy (best kernel, best termination, best n_action_steps) outperform the Columbia DP baseline on Push-T? — **CONCLUDED**: SODA k=9 (97.97%) significantly outperforms DP k=5 (89.04%), p=0.0002 (Mann-Whitney ***). SODA k=5 vs DP k=5 is not significant (p=0.47).

**Q7 (noise_study)**: Does hierarchical decomposition (SODA) improve robustness under temporally-correlated action noise? Does BID test-time sampling help DP, SODA, or both? — **IN PROGRESS**

---

## 2. Experimental Design Overview

All experiments share a fixed high policy:
```
/experiments/final_experiments/pusht/high_starts_prev_opt/best.ckpt
```
This policy was trained with multi-anchor augmentation and prev-option conditioning, achieving 92% option classification accuracy on Push-T. It is held constant throughout to isolate the effect of low-policy design choices.

### Fixed Hyperparameters (all experiments)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lr` | 1e-4 | Matches Columbia DP; loss still decreasing at epoch 500 |
| `lr_schedule_type` | cosine (kernel_sweep) / constant (termination) | kernel_sweep: 5-epoch linear warmup then cosine decay to 0 over 495 epochs. termination_study: constant at 1e-4 backbone (frozen, lr=0 effective) / 1e-3 β MLP for all 100 epochs — cosine over 100 epochs would decay LR too aggressively before β converges |
| `weight_decay` | 1e-6 | Matches Columbia DP |
| `batch_size` | 64 | Matches Columbia DP |
| `option_embed_dim` | 32 | Locked (checkpoint compatibility) |
| `bottleneck_dim` | 2048 | Locked (= down_dims[-1]) |
| `n_action_steps` | 8 | Columbia DP default; tuned in receding_horizon_study after termination_study |
| `n_obs_steps` | 2 | Two frames of visual context |
| `min_segment_len` | 3 | Drops 1- and 2-frame segments (label noise / edge cases) |
| `all_anchors` | true | Every frame in a segment is a training anchor |
| `option_balance` | none | No inverse-freq weighting on diffusion loss |
| `num_train_timesteps` | 100 | Standard DDPM |
| `β MLP hidden_dim` | 256, 2 layers | Standard capacity |
| `n_episodes (eval)` | 50 | Sufficient for stable mean/std estimates |
| `test_start_seed` | 100000 | Fixed seeds for reproducibility |
| `max_steps` | 300 | Columbia DP evaluation protocol |

### Evaluation Metric

**Mean max overlap score at 300 steps**: the maximum block-target overlap percentage achieved within 300 environment steps, averaged over 50 episodes. This is the primary metric used by Columbia DP and matches their published results.

---

## 3. Experiment 1: kernel_size_study

### Hypothesis

Larger temporal convolution kernels (k=7, 9) capture longer-range dependencies in the action sequence and may produce more coherent multi-step plans, at the cost of higher parameter count. For Push-T with relatively short option segments (~17 steps on average), the optimal kernel may differ from general imitation learning tasks.

### What Varies

`policy.kernel_size` ∈ {5, 7, 9}.

### What Is Constant

All other architecture parameters (down_dims, n_groups, etc.), training hyperparameters, horizon (auto from data), no β training. All evals use `n_action_steps=8` and duration termination.

### Methodology

1. Train k=5, k=7, and k=9 for 500 epochs each (k=7 and k=9 in parallel).
2. Epoch sweep eval: discover epoch_0050, 0100, ..., 0500 checkpoints for all three kernels.
3. Use `n_action_steps=8` and duration termination for all evals.
4. 50 episodes per checkpoint.

### Expected Outcomes

We expect k=9 (current default in soda_supervised.yaml) to perform well given our larger option-segment horizon, but k=7 may balance receptive field and overfitting risk. k=5 serves as the baseline.

### Decision Rule

The kernel with the highest best-epoch mean_score is used as the backbone for all termination_study runs. The best.ckpt from that kernel is the starting checkpoint for termination training.

> **CONCLUDED** — k=9 (epoch_0450, mean=97.97%, std=12.99%) wins over k=7 (96.69%) and k=5 (93.38%). All downstream experiments use the k=9 ep450 checkpoint.

---

## 4. Experiment 2: termination_study

The termination study has four sequential stages. All stages: 100 epochs, frozen backbone (stages 1–2) or joint (stage 3), β MLP lr=1e-3, termination_pos_weight=null (auto-calculated from dataset class frequencies, escape-relabeling-aware), label_smoothing=0.05. All evals use `n_action_steps=8`.

### Background: Why β?

Duration termination uses the model's predicted action length to decide when to switch options. This is a proxy that works when the model predicts durations accurately. A learned β head directly detects terminal states, which should be more robust to duration prediction errors and better at recognizing "I've arrived at the goal" rather than "I've run out of planned steps."

### Background: The Escape Training Signal

**Motivation**: In rollout analysis, a key failure mode is the policy getting stuck oscillating in the wrong option — e.g., locked in a push skill when the block is out of reach and repositioning is needed. β never fires because the stuck state doesn't look like a terminal push state. The issue is that β is trained only on **completion** signal (segment boundaries = option done), with no signal for **escape** (current option is inappropriate for this state).

**Reformulation**: Instead of training β to ask "is this skill complete?", we train it to ask "am I in the right skill for this state?" These are related — a terminal state of option A is also a state where option A should be escaped — but the escape formulation generalizes to mid-option stuck states that never appear as boundaries in the training data.

**Training construction**: For each training sample, with probability `p_escape=0.5`, replace the true `option_id` with a randomly sampled wrong option and set β=1. This creates examples where β must fire because the option assignment is inconsistent with the observation.

**Why option_id conditioning makes this tractable**: β already takes `option_id` as input. The question it answers is not "what option does this state belong to?" (hard — states can be visually ambiguous across options) but "given that I've been assigned option X, does this observation make sense for option X?" (easier — the option_id provides the disambiguating context). Two options that share visual states will have different option_id embeddings, and β learns to use that signal.

**Why escape relabeling is incompatible with bottleneck-based β**: At training time, a wrong-option example would use the ground-truth expert actions for the forward pass. At inference, the bottleneck is computed from the policy's own generated actions under the wrong option — which are incoherent with the state. This train/test distribution mismatch makes the learned signal unreliable. Obs-based β does not have this problem: the visual observation is identical whether actions are expert or policy-generated.

**Failure mode awareness**: Options that share highly similar visual states (e.g. pivot push vs linear push in some frames) will produce ambiguous escape labels. Label smoothing mitigates this. The hypothesis is that option_id conditioning provides enough disambiguation in practice, and that the benefit of catching stuck cases outweighs the cost of occasional early exits in ambiguous states.

---

### Stage 1a: Bottleneck Source Comparison (Frozen Backbone)

**Q1: Does DDIM bottleneck beat expert?**

**Runs**: `bottleneck_expert` vs `bottleneck_ddim_positive`
- `bottleneck_expert`: expert actions at t=0, completion-only labels
- `bottleneck_ddim_positive`: DDIM-5 generated actions, completion-only labels (closes train/inference gap)

**What is constant**: Frozen backbone, same base checkpoint, no escape relabeling, `pos_weight=null` (auto).

**Decision**: Winner → `bottleneck_best` used as the bottleneck variant in stage 2 and 3.

> **CONCLUDED** — `bottleneck_expert` wins (epoch=200, mean=88.96%) over `bottleneck_ddim_positive` (epoch=50, mean=84.11%). Expert actions outperform DDIM-generated actions for completion-only training.

---

### Stage 1b: Escape Training Signal Comparison (Frozen Backbone)

**Q2: Does positive_negative improve over positive?**

Tests whether escape relabeling (expand-all: K samples per anchor, K-1 with wrong option → β=1) improves over completion-only training. Both obs and bottleneck DDIM variants are compared.

**Runs** (3 new; `bottleneck_ddim_positive` reused from stage 1a — no retraining):
- `bottleneck_ddim_positive` — reuse from 1a
- `bottleneck_ddim_positive_negative` — DDIM escape, `pos_weight=null` (auto ≈ 0.46 for K=3)
- `obs_positive` — completion-only, `pos_weight=null` (auto ≈ 16 for L=17)
- `obs_positive_negative` — obs escape, `pos_weight=null` (auto ≈ 0.46)

**Escape relabeling (expand-all)**: each (seg, anchor) generates K samples — 1 real + K-1 wrong-option escape (β=1). Val always uses real labels only. `pos_weight` auto-formula: `(L-1)/((L-1)(K-1)+K)`.

**Why bottleneck DDIM supports escape (unlike bottleneck expert)**: With wrong option_id, DDIM generates incoherent actions matching what inference produces when stuck in the wrong option. Closes train/inference gap. Expert actions do NOT close this gap.

**Decision**: Separate winners for bottleneck and obs sides → `bottleneck_best` and `obs_best` for stage 2.

> **CONCLUDED** — `obs_positive` wins (epoch=100, mean=85.99%) over `obs_positive_negative` (74.92%) and `bottleneck_ddim_positive_negative` (78.27%). Escape relabeling degrades performance on Push-T; boundary-only training is the better signal.

---

### Stage 2: Best Combination (Frozen Backbone)

**Hypothesis**: The bottleneck and obs features are complementary — bottleneck encodes "what is the policy planning to do?", obs encodes "what does the world look like?" Combining the best from each (with the best training signal for each) may outperform either alone.

**What varies**: β MLP input and training signal:
- `bottleneck_best`: U-Net bottleneck only, boundary labels (winner from stage 1a)
- `obs_best`: ResNet(obs) + option_embed only, best labels (winner from stage 1b, either boundary or boundary+escape)
- `both`: concat([obs+option, bottleneck_best]), using obs_best training signal for the obs component

**Baseline**: Duration termination result from kernel_size_study (best kernel, best epoch). If no β variant beats this by more than 1 std, duration termination is used for comparison_study.

**Decision**: Best variant → `frozen_best` for stage 3.

> **SKIPPED** — Stage 2 was not run. Duration termination (97.97%) was established as the definitive winner after Stage 1 results; the `both` combination run was not pursued.

---

### Stage 3: Joint Training (stop_grad=False)

Repeat stages 1a, 1b, and 2 with β gradients flowing into the backbone (stop_grad=False, λ=0.01, diffusion loss active as a restoring force). This tests whether fine-tuning the backbone for termination detection improves over frozen training for each configuration.

**What varies**: `termination_stop_grad=False` applied to all frozen configurations:
- From stage 1a: `bottleneck_expert_joint`, `bottleneck_ddim_positive_joint`
- From stage 1b: `obs_positive_joint`, `obs_positive_negative_joint`
- From stage 2: `bottleneck_best_joint`, `obs_best_joint`, `both_joint`

**Runs**: `bottleneck_expert_joint`, `bottleneck_ddim_positive_joint`, `obs_positive_joint`, `obs_positive_negative_joint` (all from 1a+1b, run in parallel), then `both_joint` after 1a+1b decisions are in. Note: `bottleneck_best_joint` and `obs_best_joint` are not separate runs — they are whichever 1a/1b joint runs won, so stage 2 only adds `both_joint` as a new training session.

**Decision**: For each frozen configuration, compare joint vs frozen. If joint does not improve, use the frozen checkpoint. Best joint or frozen result → `term_best` for comparison_study.

> **CONCLUDED** — Joint training improves `bottleneck_ddim_positive` (84.11%→89.65% at ep50) but degrades `bottleneck_expert` (88.96%→87.32%). `obs_positive_joint` (84.19%) trails frozen `obs_positive` (85.99%). `obs_positive_negative_joint` (71.75%) and `bottleneck_ddim_pn_joint` (73.52%) are both worse. `both_joint` was not run. Best joint result (89.65%) still trails duration termination (97.97%) by >8 points. **Overall termination study conclusion: duration termination is the best mechanism for Push-T. All learned β variants underperform by ≥8 points and are not adopted.**

---

### Stage 4: Standalone ResNet β (Fresh ImageNet Init, Fully Trainable)

**Q5: Does a dedicated, free-to-adapt ResNet outperform a shared encoder frozen at DP pretraining?**

**Motivation**: Stages 1–3 all share the obs encoder with the diffusion backbone. Even in the joint training stage (stop_grad=False), the ResNet is initialized from the DP checkpoint and receives competing gradients from both diffusion MSE and termination BCE. Stage 4 removes this coupling entirely: a fresh ResNet (ImageNet init) owns its own weights and is trained purely to detect termination states, with no diffusion loss interfering.

**Run**: `obs_positive_negative_separate`
- Architecture: ResNet18Conv (ImageNet init, GroupNorm, trainable) + option embed + TerminationHead MLP
- Training signal: completion + escape relabeling (positive_negative, same as stage 1b `obs_positive_negative`)
- MLP capacity: hidden_dim=256, num_layers=2, dropout_rate=0.3 — identical to other term experiments
- k=9/450-epoch LowPolicy: used **only** for obs normalizer; its weights are not loaded

**What is different from `obs_positive_negative` (stage 1b)**:
- ResNet starts from ImageNet (not DP checkpoint)
- ResNet receives gradients from BCE loss (not frozen)
- Model is completely separate — no shared parameters with π_low

**Training**: 100 epochs, single Adam optimizer (lr=1e-4), checkpoint_every=10, use_ema=True.
**Eval**: Epochs [50, 100] via `rollout_hierarchical_external_beta` (50-episode rollout, beta termination mode, beta_transition=0.92). Rollout mean score is the authoritative checkpoint selection criterion — beta_val_acc_pos/neg are informational only.
**Scripts**:
- Train: `modal run --detach experiments/final_experiments/pusht/termination_study/stage4_standalone_beta/modal_train_stage4.py`
- Eval:  `modal run experiments/final_experiments/pusht/termination_study/stage4_standalone_beta/modal_eval_stage4.py`
- Results: `/experiments/final_experiments/pusht/termination_study/stage4/obs_positive_negative_separate/`

**Comparison baseline**: `obs_positive_negative` (stage 1b) — same training signal, same MLP, but obs encoder is frozen DP ResNet. If stage 4 outperforms stage 1b, a dedicated ResNet adds value beyond the shared DP representation.

> **DEFUNCT** — Stage 4 was not run. Duration termination is the confirmed winner for Push-T; a standalone ResNet β is not expected to close the ~8 point gap over duration termination. Closed out; focus moves to noise_study (Stage 5).

---

## 5. Experiment 3: receding_horizon_study

Applied to the best overall SODA policy (best kernel + best termination from stages above). This comes after termination_study because n_action_steps interacts with the termination mechanism: the optimal window for a β-gated policy may differ from duration termination.

### Hypothesis

The receding-horizon window size controls how frequently π_low replans. Too small (n=2) means frequent replanning but less action coherence; too large (n=16) means actions become stale before the policy can correct. There is an optimal value for the final SODA policy.

### What Varies

`n_action_steps` ∈ {2, 4, 6, 8, 10, 16} plus `open_loop` (execute full chunk before replan).

### What Is Constant

All other hyperparameters. Best checkpoint from termination_study (fixed) evaluated with all settings.

### Methodology

1. Reuse best.ckpt from termination_study (no additional training).
2. Evaluate with each n_action_steps value: 50 episodes, fixed seeds.

### Duration Termination Semantics (for duration-termination runs)

The low policy generates a horizon-length action chunk with a duration channel (last dim). This predicts `native_steps = round(mean_duration × horizon)`, the number of real steps the chunk represents. The executor:
- Executes `min(native_steps, n_action_steps)` steps
- If `native_steps < n_action_steps` after a replan: the option is considered complete → π_high resamples

### Expected Outcomes

We expect n_action_steps ∈ {6, 8} to perform best, consistent with Columbia DP's receding-horizon design (h=8 for Push-T). Very small values cause excessive replanning overhead; open-loop mode may lose coherence when a chunk is executed past the skill boundary.

### Decision Rule

The n_action_steps with the highest mean_score (ties broken by smaller value) is used for **comparison_study**.

> **CONCLUDED** — n=8 is optimal (mean=97.97%, std=13.00%). n=6 is the closest alternative (96.00%). Open-loop (81.65%) and large windows (n=16: 87.68%) underperform. n=8 used for comparison_study.

---

## 6. Experiment 4: comparison_study

### Hypothesis

The best SODA policy (optimal kernel, optimal termination, optimal n_action_steps) outperforms Columbia DP on Push-T because hierarchical decomposition into reposition/push/finalize skills allows each sub-policy to specialize on a simpler sub-distribution, reducing the multi-modal complexity that diffusion must capture.

### Setup

- **SODA**: best checkpoint from termination_study, evaluated with best n_action_steps from receding_horizon_study
- **DP baseline**: Columbia DP Push-T checkpoint (pusht_image_cnn_train0/latest.ckpt, 3050 epochs)
- **Fixed**: seeds, max_steps, eval protocol identical for both
- **50 episodes** per policy

### Analysis

Beyond mean score, we will examine:
- Per-episode score distribution (histogram, violin plot)
- Option sequence length and transition frequency in SODA rollouts (from overlay JSONs)
- Whether SODA fails on specific option transitions (e.g., reposition→push boundary)

> **CONCLUDED** — SODA k=9 (97.97%, std=12.99%) significantly outperforms Columbia DP k=5 (89.04%, std=22.58%), Mann-Whitney p=0.0002 (***), t-test p=0.019 (*). SODA k=5 (93.38%) vs DP k=5 (89.04%) is not significant (p=0.47). Hierarchical decomposition provides measurable performance gains at matched kernel; the larger k=9 also reduces variance substantially.

---

## 7. Eval Protocol

All evaluations use the Columbia Push-T evaluation protocol:
- **Metric**: mean max block-target overlap score (%) at 300 steps
- **Seeds**: episode i uses seed `100000 + i`, i ∈ [0, 49]
- **Env**: `PushTImageEnv`, `legacy_test=True`, `render_size=96`
- **Observation**: 2 frames (current + 1 history), 96×96 → 84×84 crop
- **Termination**: whichever termination method is under test (duration or β)
- **n_action_steps**: fixed at 8 for all runs except receding_horizon_study

---

## 8. Data and Checkpoint Management

All checkpoints, eval JSONs, and videos are stored on Modal Volume `soda-experiments`.
Local paths (`experiments/final_experiments/pusht/`) mirror the volume structure but contain
only committed scripts and downloaded results (no checkpoints tracked in git).

The `experiments/final_experiments/pusht/shared/experiment_utils.py` provides:
- `EvalResult` dataclass — common schema for all eval results
- `save_eval_result`, `load_eval_results` — JSON serialization
- `plot_eval_curve` — line plot with error bands (mean ± std vs epoch)
- `plot_bar_comparison` — bar chart for cross-run comparison

All `summary.json` files use the same schema, enabling replotting across experiments without
code changes.

---

## 9. Expected Timeline

| Study | Training Time | Eval Time | Notes |
|-------|--------------|-----------|-------|
| kernel_size_study | ~6–8 hrs (k=5/7/9, all parallel) | ~2 hrs (sweep) | First to run |
| termination_study/stage1a | ~2 hrs (2 runs parallel) | ~30 min | After Q1 answer |
| termination_study/stage1b | ~2 hrs (2 runs parallel) | ~30 min | Parallel with stage1a if budget allows |
| termination_study/stage2 | ~2 hrs (3 runs) | ~1 hr | After Q2a+Q2b answers |
| termination_study/stage3 | ~6 hrs (5 joint runs: 4 parallel then 1) | ~2 hrs | After Q3 answer |
| receding_horizon_study | 0 (reuse best ckpt) | ~1 hr (7 evals × 50 eps) | After Q4 answer |
| comparison_study | 0 (reuse ckpts) | ~1 hr (50 eps × 2) | Final |

---

## 10. Proposed Experiment: Full-State High Policy Training

### Motivation

The current π_high is trained only on segment-start observations: for each skill transition in the expert demos, it sees the state at the moment the new skill begins and learns to predict which skill was selected. This creates a **distribution mismatch at inference**: when β fires and π_high is called, the current state may not look like any segment-start state in training data — it could be a mid-skill state, a stuck state, or a state partway through an unusual trajectory. π_high has no training signal for these cases and must extrapolate blindly.

### Proposed Reformulation

Instead of learning "what skill should I start at this transition?", train π_high to answer "what skill would the expert be executing in this state?" for **any frame in the demonstrations**.

**Training data**: all (obs, option_id) pairs from the zarr — every frame from every segment, not just segment starts. The option_id label for a frame is whichever skill the expert was demonstrating at that timestep.

**Why this is better for inference**: when β fires from any state (including unusual ones), π_high has seen states that look like the current one during training. The model has learned a state→option mapping over the full state distribution, not just the thin slice of segment-start states.

**Visual ambiguity concern**: The end of reposition looks similar to the start of linear push. However, in Push-T these states are still discriminable — at the end of reposition the agent is not contacting the block, whereas at the start of linear push it is. The frozen obs encoder (or a jointly trained one) should capture this distinction via contact state and block position.

### Key Design Decisions

- **No prev_option conditioning** — the model answers "what skill is this state in?" independently, without needing to know the previous skill. This is a cleaner and more general formulation.
- **No inverse-frequency weighting** — option A having 3× more frames than option B means the expert genuinely spends 3× more time in option A. This is a real signal about `P(option | state)` and should be preserved. Reweighting would distort the learned prior, causing the model to over-predict rare options relative to their true inference-time frequency.
- **Label smoothing** — retain `label_smoothing: 0.1` (same as current π_high config) to prevent logit explosion on easy training examples and handle visually ambiguous boundary frames where the end of one skill looks like the start of the next.
- **Controlled comparison** — run with the same obs encoder checkpoint and same architecture as the current π_high, changing only the dataset sampling.

### Implementation Plan

**1. New dataset mode in `OptionStartDataset` or new `OptionAllFramesDataset`** (`soda/dataset/option_start_dataset.py` or new file):
- Load all (obs, option_id) pairs from zarr, not just segment starts
- Same train/val split by episode as current dataset
- Inverse-frequency class weights computed from frame counts per option

**2. Config flag in `high_policy` block**:
```yaml
high_policy:
  train_on_all_frames: false   # false = current behavior (segment starts only)
                               # true  = all frames, "what skill is the expert in?"
  condition_on_prev_option: false  # should be false for all-frames mode
```

**3. `train_high.py` changes**:
- Read `train_on_all_frames` flag from config
- When true: use new full-frame dataset instead of `OptionStartDataset`
- Pass per-class inverse-frequency weights to cross-entropy loss
- Assert `condition_on_prev_option: false` when `train_on_all_frames: true` (prev option is not well-defined for arbitrary frames)

**4. New experiment config** (`configs/pusht/exp_high_all_frames.yaml` or flag in `soda_supervised.yaml`):
- `train_on_all_frames: true`
- `condition_on_prev_option: false`
- Same architecture otherwise

**5. Evaluation**: run standard π_high eval (option classification accuracy on held-out segment starts) plus downstream hierarchical rollout comparison.

### Decision Rule

If all-frames π_high achieves comparable or better option classification accuracy on segment-start val frames AND improves downstream rollout scores (fewer stuck episodes, cleaner option transitions), adopt as the default training mode.

> **CONCLUDED (negative)** — all_frames π_high achieves only 77.62% downstream rollout score vs 97.97% with the segment-start policy (`high_starts_prev_opt`). Not adopted. The fixed high policy trained on segment starts with prev-option conditioning (92% option accuracy) is used throughout all experiments.

---

## 11. Anticipated Failure Modes

1. **β never fires accurately**: If β accuracy stays below 70% on val, the bottleneck/obs features are not discriminative enough for terminal-state detection. Mitigation: try higher pos_weight or lower label_smoothing.

2. **Escape relabeling causes premature exits**: If `obs_positive_negative` underperforms `obs_positive`, the wrong-option examples are causing β to fire in valid mid-option states (visual ambiguity between options outweighs the escape signal benefit). This is a valid negative result — use boundary-only training.

3. **Joint training destabilizes diffusion**: If stage 3 diffusion val_loss increases > 20% vs stage 2 starting point, λ=0.01 is too high. Mitigation: reduce to λ=0.005 and re-run.

4. **Duration termination outperforms all β variants**: If no β variant exceeds duration termination baseline + 1 std, the β head does not add value for Push-T. This is a valid negative result — report it and use duration termination for comparison_study. *(Observed — confirmed negative result.)*

5. **SODA underperforms DP**: If the best SODA policy does not match DP performance, analyze where rollouts fail (which option transitions, which episode types). The hierarchical decomposition may not benefit a task as simple as Push-T, which is also a valid scientific finding. *(Not observed — SODA k=9 significantly outperforms DP.)*

---

## 12. Experiment 5: noise_study — Noise Robustness (DP vs BID vs SODA vs SODA-BID)

**Status**: Scripts complete, runs pending (no result JSONs yet).

### Research Question (Q7)

Does hierarchical decomposition confer robustness to temporally-correlated action noise? Does BID test-time ensemble sampling provide additional robustness for DP, SODA, or both?

### Background

The poster identifies two open questions: (1) cannot isolate hierarchy from kernel scaling without a DP k=9 control, and (2) BID has not been evaluated under the same protocol as SODA/DP. The noise study addresses both by holding kernel fixed at k=5 and comparing all four policies under matched conditions.

### Noise Model

AR(1) temporally-correlated Gaussian perturbation applied to executed actions before `env.step()`, matching Liu et al. 2024 (BID paper):

```
ε_t = ρ · ε_{t-1} + η · w_t,   w_t ~ N(0, I),   ρ = 0.9
```

where η is the noise scale. Swept over η ∈ {0.0, 1.0}.

### Policies

| Tag | Description | Checkpoints |
|-----|-------------|-------------|
| `dp` | Columbia DP k=5, n_action_steps=1 | `dp_baselines/pusht_image_cnn_train0/best.ckpt` |
| `bid` | BID on DP k=5 (n_samples=16, n_mode=3, decay=0.9) | DP ckpt + `weak_policy/dp_weak/checkpoints/epoch_0015.ckpt` |
| `soda` | SODA k=5 ep350, n_action_steps=1, duration termination | `final_experiments/pusht/kernel_size_study/k5/epoch_0350.ckpt` + `high_starts_prev_opt/best.ckpt` |
| `soda_bid` | BID applied to SODA π_low (k5 strong + k5 weak reference) | Same SODA ckpts + `weak_policy/k5_weak/epoch_0015.ckpt` |

**Why k=5**: creates a fair 4-way comparison. DP k=5 is the Columbia baseline; SODA k=5 is the matched-kernel SODA. This avoids confounding noise robustness results with kernel-size differences.

**Why n_action_steps=1**: BID resamples actions at every step (it requires a full diffusion denoising pass per env step to select among proposals). Applying n_action_steps=1 uniformly across all four conditions makes the comparison fair.

### Episodes

25 per condition (extendable to 50 via `--n-episodes 50`; incremental JSON format deduplicates by seed so re-runs only add missing episodes).

### Scripts

```bash
# Run all 8 conditions (4 policies × 2 noise levels)
modal run experiments/final_experiments/pusht/noise_study/modal_eval_bid_comparison.py \
  --soda-low-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k5/epoch_0350.ckpt \
  --soda-weak-checkpoint /experiments/weak_policy/k5_weak/epoch_0015.ckpt \
  --soda-high-checkpoint /experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt

# Extend to 50 episodes
modal run experiments/final_experiments/pusht/noise_study/modal_eval_bid_comparison.py \
  --soda-low-checkpoint ... --n-episodes 50

# Plot grouped bar chart
python experiments/final_experiments/pusht/noise_study/plot_noise_study.py
```

### Expected Outcomes

1. **η=0.0**: All four policies should replicate their no-noise baselines (DP ≈89%, SODA ≈93% at k=5). BID may marginally improve DP via ensemble denoising even without noise.
2. **η=1.0**: DP is expected to degrade most; BID partially recovers DP performance via test-time ensemble selection. SODA hierarchy may confer additional robustness: when a chunk is corrupted, duration termination exits early (predicted duration shortens under noise) and π_high resamples a new option.
3. **SODA-BID** is expected to be the most robust: combines BID's per-step ensemble selection with SODA's semantic replanning.

### Decision Rule

If SODA-BID score at η=1.0 ≥ BID score + 1 std → hierarchy adds robustness beyond BID alone (positive finding). If all four degrade similarly → noise robustness is not a differentiating factor for SODA on Push-T (valid negative result).

### Output Files

```
experiments/final_experiments/pusht/noise_study/
  dp_eta0.0.json          soda_eta0.0.json
  dp_eta1.0.json          soda_eta1.0.json
  bid_eta0.0.json         soda_bid_eta0.0.json
  bid_eta1.0.json         soda_bid_eta1.0.json
  plot_noise_study.png
```

---

## 13. Experiment 6: unsupervised_study — Push-T with LOVE Option Discovery

**Status**: Not started. Awaiting LOVE label generation (Neetish).

### Research Question

Can SODA achieve comparable performance using unsupervised option labels from LOVE versus the heuristic supervised labels from E1? This directly tests whether the hierarchical structure generalizes beyond manually-designed option boundaries.

### Setup

Same training formulation as E1 (supervised Push-T), with one change: `task.dataset.option_id_key: option_id_unsupervised`. Both π_low and π_high are retrained from scratch using LOVE-generated segment labels.

**π_low**: k=5 U-Net, no β head, 500 epochs, cosine LR schedule, duration termination, checkpoint every 50 epochs. Config: `configs/pusht_unsupervised/unsupervised_k5_no_beta.yaml`.

**π_high**: segment-start observations + prev-option conditioning (same approach as the winning supervised `high_starts_prev_opt`). 1000 epochs, checkpoint every 5 epochs. Config: `configs/pusht_unsupervised/unsupervised_high_starts_prev_opt.yaml`.

**No ordering constraint**: π_low and π_high can be trained in parallel. π_high uses a fresh ResNet initialized from ImageNet weights (`imagenet_init: true`) — it does not depend on the π_low checkpoint for weight initialization.

**Why k=5**: Matches the E1 supervised baseline at the same kernel size, making the supervised vs. unsupervised comparison clean (no kernel confound).

**Why retrain π_high**: LOVE may produce a different number of options than the supervised heuristic (3). The option embedding dimension must match between π_low and π_high, so reusing the supervised `high_starts_prev_opt` checkpoint is not possible if `num_options` differs.

### Prerequisite

LOVE labels must be written into `data/raw/pusht/pusht.zarr` as `option_id_unsupervised`. See `soda/option_discovery/unsupervised/love_adapter/` and `configs/pusht_unsupervised/README.md`.

### Training Commands

```bash
# Train π_low and π_high in parallel (no ordering dependency — π_high uses ImageNet init)
modal run --detach modal/modal_train_low.py \
  --config-name unsupervised_k5_no_beta \
  --task pusht_unsupervised \
  --hydra-overrides "train_low.output_dir=/experiments/final_experiments/pusht/unsupervised_study/low train_low.wandb_run_name=unsupervised_k5_no_beta_500ep"

modal run --detach modal/modal_train_high.py \
  --config-name unsupervised_high_starts_prev_opt \
  --task pusht_unsupervised
```

### Eval Protocol

Identical to E1 / comparison_study: 50 episodes, seed=100000, max_steps=300, n_action_steps=8, duration termination. Epoch sweep over all 10 low-policy checkpoints (ep50, ep100, ..., ep500). High policy is fixed to `unsupervised_study/high/best.ckpt` throughout the epoch sweep.

### Expected Outcomes

1. If LOVE segments are semantically coherent and SODA performs within 5 points of the supervised result (≥88%), it demonstrates that unsupervised option discovery is a viable alternative to manual labeling.
2. If LOVE segments are noisy (high intra-option variance, short mean length), π_low will struggle to learn a clean option-conditioned policy and performance will lag E1 significantly.
3. Report LOVE segment statistics (num_options, mean segment length, segment length std) alongside scores to contextualize the result.

### Decision Rule

Compare best-epoch mean score for unsupervised E3 vs. supervised k=5 E1 (93.38%). If E3 ≥ E1 − 5pp: viable negative result (LOVE roughly matches supervised labeling). If E3 < E1 − 5pp: unsupervised labels produce a qualitatively weaker policy; analyze whether the gap is due to segment noise or option count mismatch.

### Output Files

```
experiments/final_experiments/pusht/unsupervised_study/
  low/
    best.ckpt
    epoch_0050.ckpt  ...  epoch_0500.ckpt
  high/
    best.ckpt
  unsupervised_results.json
  plot_unsupervised_vs_epoch_t300.png
  plot_unsupervised_vs_supervised_comparison.png
```
