"""
Standalone termination β network: fresh ResNet (ImageNet init) + option embed + MLP.

Architecture and forward pass mirror LowPolicy with termination_input=obs exactly:
  encode_obs_global_features → build_global_cond → TerminationHead(stop_grad=False)

The only differences from LowPolicy's obs β head:
  - Separate model with its own obs_encoder (not shared with diffusion)
  - ResNet initialized from ImageNet (not the DP checkpoint)
  - No stop_grad — ResNet receives gradients from BCE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from soda.models.termination_head import TerminationHead, TerminationHeadConfig


@dataclass
class StandaloneBetaConfig:
    num_options: int
    option_embed_dim: int        # 32 — must match LowPolicy
    n_obs_steps: int             # 2
    crop_shape: tuple[int, int]  # (84, 84)
    obs_encoder_group_norm: bool # True
    hidden_dim: int              # 256
    num_layers: int              # 2
    dropout_rate: float          # 0.3
    use_chunk_cursor: bool       # False


def _build_obs_encoder_and_dim(cfg: StandaloneBetaConfig, hydra_cfg: Any) -> tuple[nn.Module, int]:
    """
    Build MultiImageObsEncoder by constructing a minimal LowPolicy and extracting
    obs_encoder + obs_feature_dim. Returns (obs_encoder, obs_feature_dim_per_step).

    obs_feature_dim is read from policy.obs_feature_dim (the actual per-step output of
    the encoder, set by the DP parent class from obs_encoder.output_shape). This is the
    same value LowPolicy uses to size the termination head for termination_input=obs.
    """
    from soda.models.low_policy import LowPolicyConfig, _build_low_policy_class
    from soda.models.termination_head import TerminationHeadConfig
    from soda.training.train_low import (
        _build_noise_scheduler,
        _cfg_get,
        _get_block,
        pusht_shape_meta,
    )

    LowPolicy = _build_low_policy_class()
    policy_block = _get_block(hydra_cfg, "policy")
    down_dims = list(_cfg_get(policy_block, "down_dims", [512, 1024, 2048]))

    dummy_lp_cfg = LowPolicyConfig(
        num_options=max(cfg.num_options, 2),
        option_embed_dim=cfg.option_embed_dim,
        bottleneck_dim=int(down_dims[-1]),
        termination_head=TerminationHeadConfig(
            bottleneck_dim=int(down_dims[-1]),
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
        ),
        termination_loss_weight=1.0,
    )

    policy = LowPolicy(
        cfg=dummy_lp_cfg,
        shape_meta=pusht_shape_meta(),
        noise_scheduler=_build_noise_scheduler(hydra_cfg),
        horizon=8,
        n_action_steps=8,
        n_obs_steps=cfg.n_obs_steps,
        unet_down_dims=tuple(int(x) for x in down_dims),
        unet_kernel_size=int(_cfg_get(policy_block, "kernel_size", 9)),
        unet_n_groups=int(_cfg_get(policy_block, "n_groups", 8)),
        unet_cond_predict_scale=bool(_cfg_get(policy_block, "cond_predict_scale", True)),
        crop_shape=tuple(int(x) for x in cfg.crop_shape),
        num_inference_steps=10,
        eval_fixed_crop=bool(_cfg_get(policy_block, "eval_fixed_crop", True)),
        obs_encoder_group_norm=cfg.obs_encoder_group_norm,
        obs_as_global_cond=True,
    )

    obs_encoder = policy.obs_encoder
    # Read actual per-step feature dim from the policy (same as LowPolicy does for β head sizing)
    obs_feature_dim = int(policy.obs_feature_dim)
    del policy
    return obs_encoder, obs_feature_dim


class StandaloneBeta(nn.Module):
    """
    Standalone termination β: obs_encoder + option_embed + TerminationHead.

    Forward mirrors LowPolicy termination_input=obs:
      encode_obs_global_features(nobs) → build_global_cond(obs_feat, option_id)
      → TerminationHead(features, stop_grad=False) → sigmoid
    """

    def __init__(
        self,
        obs_encoder: nn.Module,
        option_embedding: nn.Embedding,
        term_head: TerminationHead,
        normalizer: Any,
        cfg: StandaloneBetaConfig,
        obs_feature_dim: int,
    ) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.option_embedding = option_embedding
        self.term_head = term_head
        self.normalizer = normalizer
        self.cfg = cfg
        self.obs_feature_dim = obs_feature_dim  # actual per-step encoder output dim

    @classmethod
    def build(
        cls,
        cfg: StandaloneBetaConfig,
        hydra_cfg: Any,
        normalizer: Any,
        *,
        imagenet_init: bool = True,
    ) -> StandaloneBeta:
        """
        Build StandaloneBeta, mirroring LowPolicy's obs termination head construction.

        Reads obs_feature_dim from the actual encoder (not a hardcoded constant) so
        the TerminationHead is sized correctly regardless of backbone configuration.
        """
        obs_encoder, obs_feature_dim = _build_obs_encoder_and_dim(cfg, hydra_cfg)

        if imagenet_init:
            from soda.models.low_policy import _load_imagenet_pretrained_conv_weights
            _load_imagenet_pretrained_conv_weights(obs_encoder)
            print(f"[StandaloneBeta] ImageNet weights loaded. obs_feature_dim={obs_feature_dim}")

        option_embedding = nn.Embedding(cfg.num_options, cfg.option_embed_dim)

        # Input dim mirrors obs_termination_input_dim() from low_policy.py
        input_dim = obs_feature_dim * cfg.n_obs_steps + cfg.option_embed_dim
        th_cfg = TerminationHeadConfig(
            bottleneck_dim=input_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            use_chunk_cursor=cfg.use_chunk_cursor,
            dropout_rate=cfg.dropout_rate,
        )
        term_head = TerminationHead(th_cfg)

        return cls(obs_encoder, option_embedding, term_head, normalizer, cfg, obs_feature_dim)

    def encode_obs_global_features(
        self, nobs: dict[str, Any], batch_size: int
    ) -> torch.Tensor:
        """
        Mirrors DiffusionUnetHybridImagePolicy.encode_obs_global_features exactly.

        Flattens n_obs_steps into batch dim, runs obs_encoder, reshapes back.
        nobs[key]: (B, n_obs_steps, ...) → encoder → (B, n_obs_steps * feature_dim)
        """
        # Flatten time axis into batch: (B, n_obs_steps, ...) → (B*n_obs_steps, ...)
        nobs_flat = {
            k: v[:, : self.cfg.n_obs_steps].reshape(-1, *v.shape[2:])
            for k, v in nobs.items()
            if isinstance(v, torch.Tensor)
        }
        features = self.obs_encoder(nobs_flat)  # (B*n_obs_steps, obs_feature_dim)
        return features.reshape(batch_size, -1)  # (B, n_obs_steps * obs_feature_dim)

    def build_global_cond(
        self, obs_features: torch.Tensor, option_id: torch.Tensor
    ) -> torch.Tensor:
        """Mirrors LowPolicy.build_global_cond: concat obs features with option embed."""
        opt_feat = self.option_embedding(option_id.reshape(-1).long())
        return torch.cat([obs_features, opt_feat], dim=-1)

    def forward_logit(
        self,
        obs_dict: dict[str, Any],
        option_id: torch.Tensor,
        chunk_cursor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Raw termination logit (B,) for training — no stop_grad."""
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = int(option_id.reshape(-1).shape[0])
        obs_feat = self.encode_obs_global_features(nobs, batch_size)
        features = self.build_global_cond(obs_feat, option_id)
        return self.term_head(features, stop_grad=False, chunk_cursor=chunk_cursor)

    @torch.no_grad()
    def predict_beta(
        self,
        obs_dict: dict[str, Any],
        option_id: torch.Tensor,
        chunk_cursor: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Termination probability in (0, 1) for inference."""
        if isinstance(chunk_cursor, float):
            b = int(option_id.reshape(-1).shape[0])
            chunk_cursor = torch.full(
                (b,), chunk_cursor, dtype=torch.float32, device=option_id.device
            )
        nobs = self.normalizer.normalize(obs_dict)
        batch_size = int(option_id.reshape(-1).shape[0])
        obs_feat = self.encode_obs_global_features(nobs, batch_size)
        features = self.build_global_cond(obs_feat, option_id)
        return torch.sigmoid(self.term_head(features, stop_grad=False, chunk_cursor=chunk_cursor))
