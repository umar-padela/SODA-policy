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

**Q1 (kernel_size_study)**: Among U-Net temporal kernel sizes {5, 7, 9}, which yields the best diffusion policy for Push-T option segments?

**Q2a (termination_study/stage1a)**: Does using policy-generated action plans (DDIM-5) for the bottleneck outperform using expert actions? (Train-inference distribution gap)

**Q2b (termination_study/stage1b)**: Does training obs-based β with escape relabeling (wrong-option examples → β=1) improve over boundary-only training? (Completion vs completion+escape)

**Q3 (termination_study/stage2)**: Does combining the best bottleneck source (from Q2a) with the best obs training signal (from Q2b) via feature concatenation outperform either alone?

**Q4 (termination_study/stage3)**: Does joint training (stop_grad=False, λ=0.01) improve over frozen for each of the above configurations?

**Q5 (receding_horizon_study)**: For the best overall SODA policy, what receding-horizon window size (n_action_steps) maximizes Push-T performance?

**Q6 (comparison_study)**: Does the best SODA policy (best kernel, best termination, best n_action_steps) outperform the Columbia DP baseline on Push-T?

---

## 2. Experimental Design Overview

All experiments share a fixed high policy:
```
/experiments/pusht/train_high_conditioned_prev_option/best.ckpt
```
This policy was trained with multi-anchor augmentation and prev-option conditioning, achieving 92% option classification accuracy on Push-T. It is held constant throughout to isolate the effect of low-policy design choices.

### Fixed Hyperparameters (all experiments)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lr` | 1e-4 | Matches Columbia DP; loss still decreasing at epoch 500 |
| `lr_schedule_type` | constant | Cosine over 500 epochs kills LR prematurely |
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

---

## 4. Experiment 2: termination_study

The termination study has four sequential stages. All stages: 100 epochs, frozen backbone (stages 1–2) or joint (stage 3), β MLP lr=1e-3, termination_pos_weight=10 (γ = avg_segment_len × 0.6 ≈ 17 × 0.6 ≈ 10), label_smoothing=0.05. All evals use `n_action_steps=8`.

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

---

### Stage 2: Best Combination (Frozen Backbone)

**Hypothesis**: The bottleneck and obs features are complementary — bottleneck encodes "what is the policy planning to do?", obs encodes "what does the world look like?" Combining the best from each (with the best training signal for each) may outperform either alone.

**What varies**: β MLP input and training signal:
- `bottleneck_best`: U-Net bottleneck only, boundary labels (winner from stage 1a)
- `obs_best`: ResNet(obs) + option_embed only, best labels (winner from stage 1b, either boundary or boundary+escape)
- `both`: concat([obs+option, bottleneck_best]), using obs_best training signal for the obs component

**Baseline**: Duration termination result from kernel_size_study (best kernel, best epoch). If no β variant beats this by more than 1 std, duration termination is used for comparison_study.

**Decision**: Best variant → `frozen_best` for stage 3.

---

### Stage 3: Joint Training (stop_grad=False)

Repeat stages 1a, 1b, and 2 with β gradients flowing into the backbone (stop_grad=False, λ=0.01, diffusion loss active as a restoring force). This tests whether fine-tuning the backbone for termination detection improves over frozen training for each configuration.

**What varies**: `termination_stop_grad=False` applied to all frozen configurations:
- From stage 1a: `bottleneck_expert_joint`, `bottleneck_ddim_positive_joint`
- From stage 1b: `obs_positive_joint`, `obs_positive_negative_joint`
- From stage 2: `bottleneck_best_joint`, `obs_best_joint`, `both_joint`

**Runs**: `bottleneck_expert_joint`, `bottleneck_ddim_positive_joint`, `obs_positive_joint`, `obs_positive_negative_joint` (all from 1a+1b, run in parallel), then `both_joint` after 1a+1b decisions are in. Note: `bottleneck_best_joint` and `obs_best_joint` are not separate runs — they are whichever 1a/1b joint runs won, so stage 2 only adds `both_joint` as a new training session.

**Decision**: For each frozen configuration, compare joint vs frozen. If joint does not improve, use the frozen checkpoint. Best joint or frozen result → `term_best` for comparison_study.

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

## 10. Anticipated Failure Modes

1. **β never fires accurately**: If β accuracy stays below 70% on val, the bottleneck/obs features are not discriminative enough for terminal-state detection. Mitigation: try higher pos_weight or lower label_smoothing.

2. **Escape relabeling causes premature exits**: If `obs_positive_negative` underperforms `obs_positive`, the wrong-option examples are causing β to fire in valid mid-option states (visual ambiguity between options outweighs the escape signal benefit). This is a valid negative result — use boundary-only training.

3. **Joint training destabilizes diffusion**: If stage 3 diffusion val_loss increases > 20% vs stage 2 starting point, λ=0.01 is too high. Mitigation: reduce to λ=0.005 and re-run.

4. **Duration termination outperforms all β variants**: If no β variant exceeds duration termination baseline + 1 std, the β head does not add value for Push-T. This is a valid negative result — report it and use duration termination for comparison_study.

5. **SODA underperforms DP**: If the best SODA policy does not match DP performance, analyze where rollouts fail (which option transitions, which episode types). The hierarchical decomposition may not benefit a task as simple as Push-T, which is also a valid scientific finding.
