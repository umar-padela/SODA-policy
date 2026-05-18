---
bibliography:
- reference.bib
---

::: center
**SODA: Supervised Option Discovery for Dynamic Action Chunking**\
**Team Members:** Umar Padela, Neetish Sharma\
**Emails:** umarp@stanford.edu, neetishs@stanford.edu\
:::

# Objective

The objective of this project is to demonstrate that the performance of
action chunking approaches can be improved by transitioning from
fixed-horizon action chunks to semantically-grounded, variable-horizon
chunks. Standard action chunking methods---including open-loop, receding
horizon control, and bidirectional decoding---operate on a fixed
temporal window (e.g., $h = 50$ steps). Unlike humans, who naturally
plan in terms of meaningful sub-goals and intent-driven outcomes, these
models are restricted to planning $h$ steps ahead. We propose that
action chunks should be semantic units (options) rather than arbitrary
temporal ones. Instead of the model asking \"What are the next 50
steps?\", it should reason about how to complete a meaningful option
(e.g., a \"Grasp\" or a \"Push\") and predict a task-dependent horizon
for that specific intent. We hypothesize that a trajectory generated
with an understanding of the trajectory's end-goal will be fundamentally
more robust and efficient than one generated over a fixed window.
Because self-supervised learning of options is difficult in practice, we
propose weak-supervision through text annotations of video via video
language models. [To isolate the contribution of the VLM supervision
signal itself, we additionally evaluate a fully unsupervised option
discovery baseline that learns segmentations directly from demonstration
data without any language input.]{style="color: black"} To evaluate
this, we plan to compare the sucess rates of a novel semantic, option
discovery, variable-horizon action chunking policy to baseline
fixed-horizon action chunking policies (vanilla open-loop, vanilla
closed-loop, receding-horizon closed-loop) [as well as a hierarchical IL
policy trained with unsupervised option discovery]{style="color: black"}
on the Push-T and Can simulation tasks, under varying amounts of
stochasticity.

This objective is important because it addresses the fundamental
limitation of rigid, fixed-horizon action chunking while solving the
long-standing challenge of option discovery through the use of
VLM-driven weak supervision. By grounding robotic actions in semantic
intent, we facilitate a shift toward policies that more closely resemble
human cognitive planning, where the physical completion of a sub-goal
dictates the transition to the next action rather than an arbitrary
clock. Establishing this framework for semantic chunks opens a promising
avenue for integrating bidirectional decoding into the option discovery
process, potentially enabling the model to refine the coherence of its
plans to further improve performance and precision.

# Related Work

## Temporal Abstraction and the Options Framework

The foundational theoretical framework for this work is the *options
framework*, introduced by [@framework_for_temporal_abstraction_in_RL],
which extends standard Markov Decision Processes (MDPs) to Semi-Markov
Decision Processes (SMDPs). Options are defined as closed-loop policies
$\pi$ that execute for a variable duration until a termination condition
$\beta$ is met [@framework_for_temporal_abstraction_in_RL]. While this
framework allows for reasoning at multiple levels of temporal
abstraction, traditionally, discovering these options autonomously has
remained a challenge [@755_option_discovery_using_deep_skill_chaining].
Recent advancements such as *Deep Skill Chaining* have sought to
discover options in high-dimensional continuous domains, but often rely
on sparse rewards or specific environmental interactions rather than the
semantic structures inherent in demonstration data
[@755_option_discovery_using_deep_skill_chaining].

## Action Chunking and Generative Policies

Modern imitation learning has shifted toward generative models equipped
with *action chunking*, such as the Action Chunking Transformer (ACT)
and Diffusion Policy [@BID_paper]. These methods predict a sequence of
future actions to capture temporal dependencies and handle multi-modal
distributions in human demonstrations [@BID_paper]. However,
fixed-horizon action chunking approaches suffer from a
consistency-reactivity tradeoff. To address this trade-off,
*Bidirectional Decoding* (BID) was proposed as a test-time inference
algorithm that uses backward coherence and forward contrast to search
for optimal action chunks across replanning steps [@BID_paper]. While
BID improves reactivity and smoothness, it remains constrained by the
fixed-horizon nature of the underlying policy [@BID_paper].

## Variable-Horizon Planning

The limitation of fixed planning horizons is explicitly addressed in
*Reinforcement Learning for Flow-Matching Policies* by Pfrommer et al.,
which introduces a horizon channel into the flow-matching velocity field
[@RL_for_flow_matching_policies]. This allows the model to generate
variable-duration trajectories by interpolating action chunks to a fixed
latent buffer and predicting the original horizon as an additional
channel [@RL_for_flow_matching_policies]. Our work builds on this
\"temporal stretching\" mechanism but shifts the focus toward semantic
sub-tasks discovered through imitation learning rather than purely
minimum-time RL objectives [@RL_for_flow_matching_policies].

## Skill Discovery and Weak Supervision

Unsupervised skill discovery methods like *CompILE* use a differentiable
sequence segmentation module to learn latent codes for variable-length
segments [@compositional_imitation_learning_and_exec]. Other approaches,
such as *LOVE*, use compression-based objectives to identify statistical
regularities in demonstrations [@learning_options_via_compression][,
framing skill discovery as minimum description length over the latent
skill sequence so that degenerate solutions (skills of length one, or
one skill per trajectory) are penalized.]{style="color: black"} However,
purely unsupervised methods can struggle to align with
human-interpretable sub-tasks
[@compositional_imitation_learning_and_exec][, and the discovered
segments are often optimized for reconstruction rather than for the
boundaries a downstream controller would benefit from. We take this as
motivation rather than disqualification: an unsupervised method gives us
a strong reference point for what option discovery looks like without
any external grounding signal, and the gap between it and the
VLM-supervised variant directly measures the value of the language
supervision.]{style="color: black"}

Weakly-supervised approaches provide a middle ground. *TACO* utilizes
task sketches---sequences of sub-tasks without boundary annotations---to
align demonstrations with sub-policies
[@learning_task_decomp_via_temporal_alignment_for_control]. Similarly,
*Modular Multitask RL with Policy Sketches* uses named sub-tasks to
guide the learning of modular sub-policies
[@Modular_Multitask_Reinforcement_Learning_with_Policy_Sketches]. Our
proposed framework extends these ideas by utilizing Vision-Language
Models (VLMs) to provide the weak supervision necessary to segment
demonstrations into semantic options, directly resolving the
\"blindness\" of fixed-horizon baselines.

# Technical Outline

## Overview

We propose a hierarchical imitation learning policy where the high-level
policy $\pi_\textbf{high}$ selects from a set of options
$\omega \in \Omega$, and a low-level policy $\pi_{\textbf{low}}$,
conditioned on the current state $s$ and option $\omega$, generates a
variable-horizon action chunk that spans the entire duration of the
selected option. Each option consists of:

- A shared low-level policy $\pi_{\textbf{low}}(a|s, \omega)$, which
  generates an action chunk

- A termination function $\beta_{\omega}(s) \in [0,1]$, which determines
  the probability of the option ending at state $s$.

The high-level policy selects an option $\omega$, and passes control
over to the low-level policy. The low-level policy generates a
variable-horizon action chunk that spans the entire duration of the
option, at each time step, in a receding-horizon manner, until the
termination condition is met, at which point the high-level policy
selects another option and the process repeats.

The high-level policy $\pi_\textbf{high}(\omega|s)$ is trained via
imitation learning with flow matching loss on expert demonstrations to
perform option selection. Given the current state $s$, the high-level
policy outputs a distribution over the discrete option space $\Omega$.

The low-level policy $\pi_\textbf{low}$ is built from a diffusion policy
backbone [@diffusionpolicy], with modifications described in the
following sections for variable-horizon action chunking and reactive
termination.

## Weakly-Supervised Option Discovery

To discover options without manual labeling, we utilize Vision-Language
Models (VLMs) to provide weak supervision on expert demonstrations.
Given an expert demonstration, the VLM identifies semantic change
points. For example, given an expert demonstration of the push-T task, a
VLM can be used to identify boundaries between options like
\"reposition\", \"pivot-push\", and \"linear-push\".

These identified options $\omega \in \Omega$ serve as the ground truth
for our termination condition $\beta_\omega$, and internal policy
$\pi_\textbf{low}(a | s, \omega)$ . By segmenting the dataset into
variable-length trajectories $h_i$, we can train the model to recognize
the inherent structure of the task.

## Unsupervised Option Discovery Baseline

To isolate the contribution of the VLM supervision signal, we train a
parallel hierarchical IL policy whose option boundaries and option
labels come from an unsupervised skill discovery method rather than a
VLM. Concretely, we plan to use *LOVE*
[@learning_options_via_compression] as the discovery backbone: it learns
a discrete option set $\Omega'$ and a termination policy by jointly
maximizing the ELBO and minimizing the description length of the latent
skill sequence over the demonstration set, which empirically avoids the
degenerate trivial-skill solutions that pure max-likelihood objectives
admit. Public code exists, which keeps the baseline tractable on our
compute budget.

We then plug the discovered options into the same hierarchical
architecture used for the VLM-supervised method (Sec. 3.1), so the only
thing that changes between the two pipelines is where the segmentation
comes from. This gives us a clean A/B: same low-level diffusion
backbone, same termination head structure, same training regime,
different supervision source. We expect the unsupervised variant to
underperform the VLM-supervised one, particularly on tasks where
semantic sub-goals don't align cleanly with statistical regularities
(e.g., the \"reposition\" phase of Push-T can look like idle or noise in
the action stream), but the magnitude and direction of that gap is
itself the result we care about. If the unsupervised variant is
competitive, that's a meaningful finding too --- it would suggest the
structural prior from variable-horizon chunking matters more than the
supervision signal.

Practically, \"tweaks to get it to work\" likely involves: (i) adapting
LOVE's input encoder for image-based observations on Push-T / Can rather
than the gridworld setting it was developed in, (ii) tuning the
description-length weight $\lambda$ and the minimum skill length, and
(iii) filtering the learned option set down to the ones with non-trivial
marginal probability before passing them to the hierarchical controller.

## Variable-Horizon Action Chunking via Temporal Stretching

A primary challenge in learning from variable-length segments is the
requirement for fixed-length input and output tensors in standard neural
architectures. To address this, we extend the base diffusion policy
backbone [@diffusionpolicy] via a *Temporal Stretching* technique. This
approach treats the intended duration of a movement as a learnable
feature, implemented through the following three-step pipeline:

1.  **Canonical Resampling:** During training, expert demonstrations of
    varying length $h_i$ are linearly interpolated to a fixed maximum
    length $h_{max}$. This ensures the 1D U-Net receives a consistent
    tensor shape while preserving the full geometric profile of the
    trajectory.

2.  **Action-Horizon Augmentation:** We augment the action output space
    by adding a dedicated *Horizon Channel*. Instead of predicting only
    the $D$ dimensions of robot movement, the architecture is modified
    to predict $D+1$ dimensions, where the final channel is populated
    with the normalized duration of the expert's original demonstration.

3.  **Dynamic Decoding:** During inference, the policy jointly denoises
    the action sequence and the horizon signal. The agent self-evaluates
    the optimal timing by decoding the $D+1$ channel---typically via
    mean-pooling across the prediction horizon---and resamples the
    generated action chunk down to that predicted duration $h$ for
    real-time execution.

## Reactive Option Termination via Binary Classification

We secondly extend the architecture by integrating a learned *option
termination head* alongside the primary diffusion decoder. While the
base 1D U-Net is optimized for generative trajectory prediction, we
attach a secondary classification head to the U-Net's *global feature
bottleneck*. This bottleneck serves as the model's central latent
representation, capturing the distilled visual and proprioceptive
features necessary to understand the current task state.

This implementation is defined by the following technical components:

- **Binary Classification Head:** We utilize a Multi-Layer Perceptron
  (MLP) that branches from the shared latent features. This head maps
  the environment state to a single scalar value
  $\beta_\omega \in [0, 1]$, representing the probability that the
  active skill has reached its semantic goal.

- **Supervised Training:** During the training phase, every frame of the
  expert demonstrations is assigned a binary label (0 or 1). The
  termination head is optimized using a *Binary Cross-Entropy (BCE)
  loss*, teaching the network to distinguish between progress-heavy
  states and successful completion states.

- **Thresholded Inference:** During real-time execution, this head
  provides a per-step monitoring signal. The high-level controller
  treats a skill as complete only when the predicted probability exceeds
  a predefined confidence threshold $\beta_{\text{transition}}$,
  allowing for reactive and asynchronous transitions between different
  robotic options.

By decoupling the termination logic from a fixed timer, the policy
becomes robust to external perturbations. If a task is completed earlier
than expected due to environment dynamics, the termination head triggers
an immediate transition, significantly improving the temporal efficiency
of the overall system.

## Loss Function

We optimize the low-level policy using a joint training regime that
combines the generative diffusion loss with the discriminative
termination loss into a single multi-task objective:
$L_\textbf{low} = L_\textbf{diffusion} + \lambda L_\textbf{termination}$.
Because the binary classification of the termination head typically
converges faster than the high-dimensional trajectory regression of the
diffusion backbone, we introduce a weighting hyperparameter $\lambda$.
This coefficient is tuned via loss magnitude matching, ensuring the
gradients from the termination head do not overpower the structural
learning of the action chunks. This joint optimization allows the global
feature bottleneck to develop a shared representation that is
simultaneously aware of the motor skills required to execute an option
and the visual features that signal its completion. This area represents
a potential challenge, and is one where we would like feedback.

## Novel Technical Contribution

The core innovation of this work lies in the **unprecedented integration
of semantic option discovery with a variable-horizon diffusion policy**.
While standard action chunking methods (e.g., ACT, Diffusion Policy)
rely on a fixed, manually-tuned hyperparameter $h$, our approach is the
first to:

1.  **Endogenize the Horizon:** Instead of treating chunks as arbitrary
    temporal slices, we treat them as complete semantic sub-tasks where
    the horizon is a learned, state-dependent variable.

2.  **VLM-Guided SMDP Discovery:** We utilize the reasoning capabilities
    of VLMs to bootstrap the discovery of options, bridging the gap
    between high-level linguistic task decomposition and low-level
    continuous control. This approach has not been previously combined
    with diffusion-based action chunking.

3.  **Elastic Motion Representation:** The use of Temporal Stretching
    allows a single diffusion model to represent variable-duration
    behaviors in a unified latent space, resolving the fundamental
    incompatibility between fixed-dimension neural outputs and
    variable-length real-world actions.

## Experimental Design and Baseline Comparison

To evaluate the efficacy of our proposed architectural modifications, we
conduct a comparative analysis against the base *Diffusion Policy*
[@diffusionpolicy]. We utilize official, checkpoints provided by the
original authors as a performance benchmark for the standard 1D U-Net
backbone on the **Push-T** and **Can** tasks. We additionally compare
against the unsupervised option discovery variant described in Sec. 3.3,
which shares our hierarchical architecture but receives its option
boundaries from LOVE rather than a VLM. The baseline set is therefore:
(i) vanilla Diffusion Policy under each of the three control regimes
below, (ii) our full VLM-supervised method, and (iii) our
unsupervised-discovery ablation.

### Standardized Training Protocol

To ensure a rigorous comparison, we adopt a *Sample-Budget Matching*
strategy. Our model is trained on an identical dataset of expert
demonstrations and for exactly the same number of epochs as the base
diffusion policy checkpoint that we use as our baseline. While the
training budget remains constant, we permit task-specific adjustments to
the **Learning Rate** and **Optimizer weight decay**. This is necessary
as the integration of the *Termination Head* and *Horizon Channel*
introduces new gradient dynamics that may require different tuning than
the vanilla generative-only backbone to reach convergence.

### Control Regimes for Evaluation

We evaluate our model against the baseline under three distinct
execution regimes to measure the trade-offs between plan stability and
reactivity:

- **Open-Loop Action Chunking:** The model predicts a full action
  sequence and executes all $H$ steps without visual re-observation.

- **Closed-Loop A($T_a=1$):** The policy re-observes the environment and
  re-plans at every environment step.

- **Receding Horizon ($T_a=8$):** The standard benchmark mode where a
  fixed subset of the predicted chunk is executed before re-planning.

### Evaluation Metrics

Performance is quantified using two primary metrics:

1.  **Success Rate:** The percentage of episodes where the robot
    completes the task (e.g., T-block within the goal zone or can placed
    on the saucer) within a maximum time limit.

2.  **Time to Completion (TTC):** For successful trials only, we record
    the total number of environment steps required. This metric is
    critical for evaluating the efficiency gains provided by our
    *Temporal Stretching* and *Variable-Horizon* execution.

## Discussion on Benchmarking Fairness

A significant challenge in this research is establishing a truly "fair"
comparison when the underlying architecture of the proposed model
differs from the baseline. While we have matched training data and epoch
counts, we would like feedback on whether there are additional
parameters that must be matched. With regards to choosing a checkpoint
to compare against, we plan to estimate a reasonable number of epochs we
can train for within our computing budget, and choose a reasonable
checkpoint accordingly.

# Team Contributions

- **Umar Padela**: label Push-T dataset via VLM, augment base diffusion
  policy with temporal stretching and reactive option termination
  architectural changes, train a policy for push-T task, and compare to
  base diffusion policy.

- **Neetish Sharma**: scope and implement the unsupervised option
  discovery baseline (LOVE-based hierarchical IL on the Can dataset,
  including encoder adaptation for image observations and tuning the
  compression objective), train the policy for the Can task, and run the
  comparison against vanilla Diffusion Policy and against Umar's
  VLM-supervised variant.
