"""
Low-level option-conditioned diffusion policy π_low(a | s, ω) + duration channel.

Extends Columbia Diffusion Policy (subclass in ``soda/`` only — project_plan §4.1)
with:
- **Option conditioning:** ``nn.Embedding(ω)`` concatenated to global cond (not raw int).
- **D+1 actions:** last channel = normalized segment duration target for ``TemporalStretcher``.
- **Termination:** ``TerminationHead(stop_grad(U-Net bottleneck))`` — BCE, no grad into U-Net.

Not used for frozen vanilla DP baseline eval; SODA hierarchical rollouts use this + π_high.

Upstream hook points (``third_party/diffusion_policy``)
---------------------------------------------------------
- ``diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py``
      ``DiffusionUnetHybridImagePolicy`` — Push-T hybrid (vision + proprio); **subclass this**.
- ``diffusion_policy/policy/diffusion_unet_image_policy.py``
      ``DiffusionUnetImagePolicy`` — image-only tasks (Square); subclass if needed later.
- ``diffusion_policy/model/diffusion/conditional_unet1d.py``
      ``ConditionalUnet1D`` — may need **bottleneck expose** (return mid features).
- ``diffusion_policy/workspace/train_diffusion_unet_hybrid_image_workspace.py``
      Training loop pattern for ``soda/training/train_low.py``.

References
----------
- ``project_plan.md`` §2 (π_low), §4.1–§4.2, §8 (locked: stop_grad, mean-pool horizon)
- ``soda/dataset/option_aware_dataset.py`` — ``OptionLabeledZarrDataset``, ``beta_label``
- ``soda/dataset/temporal_stretch.py`` — stretch targets; mean-pool duration decode at infer
- ``soda/models/termination_head.py``, ``soda/training/losses_low.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from soda.models.termination_head import TerminationHead, TerminationHeadConfig

if TYPE_CHECKING:
    # Import after ``soda.eval.policy_loaders._require_diffusion_policy()`` at runtime.
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LowPolicyConfig:
    """
    SODA-specific hyperparameters (Hydra ``low_policy:`` block).

    DP backbone fields (``horizon``, ``n_obs_steps``, U-Net dims, noise scheduler, …)
    stay in the upstream Hydra config / checkpoint — mirror ``baseline_vanilla.yaml``
    and extend ``shape_meta`` action dim to **D+1**.

    All fields here required — no silent defaults (``configs/pusht/soda_*.yaml``).
    """

    num_options: int
    option_embed_dim: int
    horizon_stretch_max: int
    # Must match U-Net mid dim after upstream exposes bottleneck (infer from DP unet or cfg).
    bottleneck_dim: int
    termination_head: TerminationHeadConfig
    termination_loss_weight: float


# ---------------------------------------------------------------------------
# Low policy (subclass DP)
# ---------------------------------------------------------------------------


class LowPolicy:
    """
    π_low: DP diffusion U-Net + option embed + duration channel + termination head.

    Integration strategy (locked — project_plan §4.1)
    -------------------------------------------------
    1. **Subclass** ``DiffusionUnetHybridImagePolicy`` (do not copy the whole policy).
    2. **Reuse** DP ``compute_loss`` / ``predict_action`` / noise scheduler / normalizer.
    3. **Override** only:
       - ``__init__`` — ``option_embed``, ``TerminationHead``, action dim D+1 in model.
       - ``compute_loss`` — call super, add ``L_termination`` via ``losses_low``.
       - ``predict_action`` (optional) — decode duration channel → ``TemporalStretcher.unstretch``.
       - helper to build **global_cond** = ``concat(DP_obs_features, option_embed(ω))``.
    4. **Bottleneck hook** — one of:
       - Minimal change to upstream ``ConditionalUnet1D.forward`` to return ``(out, bottleneck)``;
       - Or override the inner ``model`` call in this class to capture mid features.

    Training batch (from ``OptionLabeledZarrDataset``)
    ---------------------------------------------------
    Expects DP-compatible keys plus SODA fields, e.g.::

        obs: {image, agent_pos, ...}   # normalized like DP
        action: (B, T, D+1)            # stretched chunk; last dim = duration target
        option_id: (B,) long
        beta_label: (B,) float {0,1}   # anchor frame (segment end)

    Loss (``losses_low.low_policy_total_loss``)::

        L_total = L_diffusion + w * L_termination

    Inference (``HierarchicalController`` — later)
    ----------------------------------------------
    - Condition on current ω from π_high.
    - ``predict_action`` → action chunk; mean-pool duration channel → native steps.
    - Step env until ``sigmoid(termination_head(bottleneck.detach())) > beta_transition``.
    """

    def __init__(
        self,
        cfg: LowPolicyConfig,
        dp_policy: "DiffusionUnetHybridImagePolicy",
    ) -> None:
        """
        Wrap or subclass instance built from DP Hydra cfg / checkpoint.

        TODO §7 row 18
        --------------
        - [ ] Prefer ``class LowPolicy(DiffusionUnetHybridImagePolicy)`` and call ``super().__init__(...)``
              with modified ``shape_meta`` (action shape last dim = Da+1).
        - [ ] ``self.cfg = cfg``
        - [ ] ``self.option_embed = nn.Embedding(cfg.num_options, cfg.option_embed_dim)``
        - [ ] ``self.termination_head = TerminationHead(cfg.termination_head)``
        - [ ] Verify ``global_cond_dim`` increased by ``option_embed_dim`` in U-Net / policy.
        - [ ] Register modules so ``state_dict`` covers SODA + DP weights for checkpointing.
        """
        raise NotImplementedError("TODO §7 row 18: LowPolicy.__init__")

    # ------------------------------------------------------------------
    # Option conditioning
    # ------------------------------------------------------------------

    def option_embedding(self, option_id: torch.Tensor) -> torch.Tensor:
        """
        Map discrete ω → vector for global conditioning.

        Parameters
        ----------
        option_id
            ``(B,)`` long

        Returns
        -------
        embed
            ``(B, option_embed_dim)``

        TODO §7 row 18
        --------------
        - [ ] ``return self.option_embed(option_id)``
        """
        raise NotImplementedError("TODO §7 row 18: option_embedding")

    def build_global_cond(
        self,
        obs_features: torch.Tensor,
        option_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concat DP observation features with option embedding (locked §8).

        Parameters
        ----------
        obs_features
            ``(B, obs_feat_dim)`` from ``obs_encoder`` (same layout as frozen DP global cond).
        option_id
            ``(B,)`` long

        Returns
        -------
        global_cond
            ``(B, obs_feat_dim + option_embed_dim)``

        TODO §7 row 18
        --------------
        - [ ] ``return torch.cat([obs_features, self.option_embedding(option_id)], dim=-1)``
        - [ ] Wire into U-Net forward wherever DP passes ``global_cond`` today.
        """
        raise NotImplementedError("TODO §7 row 18: build_global_cond")

    # ------------------------------------------------------------------
    # U-Net bottleneck (for termination)
    # ------------------------------------------------------------------

    def forward_unet_with_bottleneck(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run conditional U-Net; return prediction + mid bottleneck for β head.

        Parameters
        ----------
        sample
            Noisy action trajectory ``(B, T, Da+1)`` (or flattened per DP convention).
        timestep
            Diffusion timestep ``(B,)``.
        global_cond
            ``build_global_cond`` output.

        Returns
        -------
        model_out, bottleneck
            ``model_out`` — same as DP noise/sample prediction.
            ``bottleneck`` — ``(B, bottleneck_dim)`` mid U-Net features.

        TODO §7 row 18 + upstream ``conditional_unet1d.py``
        -----------------------------------------------------
        - [ ] After upstream exposes bottleneck: ``out, bn = self.model(..., return_bottleneck=True)``
        - [ ] Document exact tensor shape from DP mid_modules (depends on ``down_dims``).
        """
        raise NotImplementedError("TODO §7 row 18: forward_unet_with_bottleneck")

    def predict_termination_logit(
        self,
        bottleneck: torch.Tensor,
    ) -> torch.Tensor:
        """
        β head on **stopped** features (no grad into U-Net — project_plan §8).

        TODO §7 row 18
        --------------
        - [ ] ``return self.termination_head(bottleneck.detach())``
        """
        raise NotImplementedError("TODO §7 row 18: predict_termination_logit")

    # ------------------------------------------------------------------
    # Loss (training)
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """
        DP diffusion loss + termination BCE (delegates to ``losses_low``).

        Parameters
        ----------
        batch
            Collated ``OptionLabeledZarrDataset`` sample (DP keys + ``option_id``, ``beta_label``).

        Returns
        -------
        loss
            Scalar ``L_total`` for backward.
        log_dict
            Detached floats for W&B (``loss_diffusion``, ``loss_termination``, …).

        TODO §7 row 18
        --------------
        - [ ] Build ``global_cond`` with ``option_id`` from batch.
        - [ ] Run U-Net forward; get ``model_out`` + ``bottleneck``.
        - [ ] ``L_diffusion`` — reuse DP logic (override super ``compute_loss`` or extract shared helper).
        - [ ] ``L_term`` — ``termination_bce_loss(predict_termination_logit(bn), batch['beta_label'])``
        - [ ] ``return low_policy_total_loss(L_diffusion, L_term, termination_weight=cfg...)``
        - [ ] Ensure **only** ``L_diffusion`` backprops through U-Net; termination path uses ``detach``.
        """
        raise NotImplementedError("TODO §7 row 18: compute_loss")

    # ------------------------------------------------------------------
    # Inference (duration decode + DP predict_action)
    # ------------------------------------------------------------------

    @staticmethod
    def decode_segment_steps(
        action_chunk: torch.Tensor,
        horizon_stretch_max: int,
    ) -> float | torch.Tensor:
        """
        Mean-pool duration channel → predicted native segment length (locked §2).

        Parameters
        ----------
        action_chunk
            ``(T, Da+1)`` or ``(B, T, Da+1)`` — last index is duration target in [0, 1].
        horizon_stretch_max
            Training stretch cap from config.

        Returns
        -------
        segment_steps_pred
            Scalar or ``(B,)`` float — pass to ``TemporalStretcher.unstretch``.

        TODO §7 row 18
        --------------
        - [ ] ``duration = action_chunk[..., -1].mean(dim=-1)``  # mean over time axis
        - [ ] ``return duration * horizon_stretch_max`` (clamp/min 1 if needed)
        """
        raise NotImplementedError("TODO §7 row 18: decode_segment_steps")

    def predict_action(
        self,
        obs_dict: dict[str, Any],
        option_id: torch.Tensor,
    ) -> dict[str, Any]:
        """
        DP-style action prediction conditioned on ω.

        TODO §7 row 18
        --------------
        - [ ] Inject ``option_id`` into global cond (same as training).
        - [ ] Call parent ``predict_action`` / internal sampler with modified cond.
        - [ ] Optionally attach ``segment_steps_pred`` from ``decode_segment_steps`` for controller.
        - [ ] Return dict compatible with ``pusht_rollout`` (``action`` key shape ``(B, Ta, Da)``).
        """
        raise NotImplementedError("TODO §7 row 18: predict_action")

    def reset(self) -> None:
        """Clear DP observation cache between rollouts (delegate to parent)."""
        raise NotImplementedError("TODO §7 row 18: reset")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dp_config(
        cls,
        cfg: LowPolicyConfig,
        dp_cfg: Any,
    ) -> LowPolicy:
        """
        Construct from Hydra DP config with SODA fields patched in.

        TODO §7 row 18
        --------------
        - [ ] Bump ``shape_meta.action.shape[-1]`` by 1 for duration channel.
        - [ ] Instantiate ``DiffusionUnetHybridImagePolicy`` via ``hydra.utils.instantiate``.
        - [ ] Wrap as ``LowPolicy(cfg, dp_policy)``.
        """
        raise NotImplementedError("TODO §7 row 18: LowPolicy.from_dp_config")

    @classmethod
    def from_checkpoint(
        cls,
        cfg: LowPolicyConfig,
        checkpoint: str | Path,
        device: str = "cpu",
    ) -> LowPolicy:
        """
        Load DP weights from ``.ckpt`` and attach SODA heads (train resume / finetune).

        TODO §7 row 18
        --------------
        - [ ] Reuse ``soda.eval.policy_loaders.load_dp_image_policy_and_cfg`` pattern.
        - [ ] Load strict=False for new ``option_embed`` + ``termination_head`` keys if needed.
        """
        raise NotImplementedError("TODO §7 row 18: LowPolicy.from_checkpoint")
