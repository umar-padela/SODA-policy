"""
SODA π_low for state-based (lowdim) tasks — LowdimSODAPolicy.

Extends Columbia DiffusionUnetLowdimPolicy with:
  - Option conditioning: nn.Embedding(ω) concatenated to global_cond
  - Termination head: MLP on [obs_flat, option_embed] → scalar β logit
  - predict_action accepts obs_dict['option_id'] in addition to obs_dict['obs']

Architecture
------------
  global_cond = cat([obs_flat, option_embed], dim=-1)   # (B, T_obs*obs_dim + option_embed_dim)
  β_input     = cat([obs_flat, option_embed], dim=-1)   # same vector
  β_logit     = termination_head(β_input)               # (B,)

Unlike the image policy, there is no ResNet encoder — obs is the raw state vector.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def _ensure_dp_path() -> None:
    from soda.eval.policy_loaders import _ensure_diffusion_policy_path
    _ensure_diffusion_policy_path()


@contextmanager
def _widen_lowdim_global_cond(extra: int):
    """Temporarily widen ConditionalUnet1D global_cond_dim by ``extra`` during __init__."""
    import diffusion_policy.policy.diffusion_unet_lowdim_policy as _mod
    orig = _mod.ConditionalUnet1D

    class _Wider(orig):
        def __init__(self, *a, global_cond_dim=None, **kw):
            if global_cond_dim is not None:
                global_cond_dim = int(global_cond_dim) + extra
            super().__init__(*a, global_cond_dim=global_cond_dim, **kw)

    _mod.ConditionalUnet1D = _Wider
    try:
        yield
    finally:
        _mod.ConditionalUnet1D = orig


class LowdimSODAPolicy(nn.Module):
    """
    State-based SODA π_low.  Wraps DiffusionUnetLowdimPolicy with option conditioning.

    Parameters
    ----------
    dp_policy_kwargs : dict
        All kwargs for DiffusionUnetLowdimPolicy.__init__ (horizon, obs_dim, action_dim, …).
    num_options : int
        Number of discrete options.
    option_embed_dim : int
        Dimension of option embedding concatenated to global_cond.
    termination_hidden_dim : int
        Hidden dim of the 2-layer β MLP.
    """

    def __init__(
        self,
        dp_policy_kwargs: dict,
        num_options: int,
        option_embed_dim: int,
        termination_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        _ensure_dp_path()
        from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
            DiffusionUnetLowdimPolicy,
        )

        # Widen U-Net global_cond_dim to accommodate option embedding
        with _widen_lowdim_global_cond(option_embed_dim):
            self.dp_policy = DiffusionUnetLowdimPolicy(**dp_policy_kwargs)

        self.option_embed = nn.Embedding(num_options, option_embed_dim)
        self.num_options = num_options
        self.option_embed_dim = option_embed_dim

        # β MLP: [obs_flat + option_embed] → scalar logit
        obs_dim        = dp_policy_kwargs["obs_dim"]
        n_obs_steps    = dp_policy_kwargs["n_obs_steps"]
        beta_input_dim = obs_dim * n_obs_steps + option_embed_dim

        self.termination_head = nn.Sequential(
            nn.Linear(beta_input_dim, termination_hidden_dim),
            nn.ReLU(),
            nn.Linear(termination_hidden_dim, termination_hidden_dim),
            nn.ReLU(),
            nn.Linear(termination_hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Forwarding properties to the wrapped DP policy
    # ------------------------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def set_normalizer(self, normalizer):
        self.dp_policy.set_normalizer(normalizer)

    def get_normalizer(self):
        return self.dp_policy.normalizer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_global_cond(self, obs: torch.Tensor, option_id: torch.Tensor) -> torch.Tensor:
        """
        obs:       (B, T_obs, obs_dim)
        option_id: (B,)  long
        returns:   (B, T_obs*obs_dim + option_embed_dim)
        """
        B = obs.shape[0]
        nobs = self.dp_policy.normalizer["obs"].normalize(obs)
        nobs_flat = nobs[:, :self.dp_policy.n_obs_steps].reshape(B, -1)
        opt_emb = self.option_embed(option_id)                  # (B, E)
        return torch.cat([nobs_flat, opt_emb], dim=-1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_action(self, obs_dict: dict) -> dict:
        """
        obs_dict:
          'obs'       — (B, T_obs, obs_dim)
          'option_id' — (B,) long
        """
        obs       = obs_dict["obs"]
        option_id = obs_dict["option_id"]

        nobs = self.dp_policy.normalizer["obs"].normalize(obs)
        B    = nobs.shape[0]
        T    = self.dp_policy.horizon
        Da   = self.dp_policy.action_dim

        global_cond = self._build_global_cond(obs, option_id)

        shape     = (B, T, Da)
        cond_data = torch.zeros(shape, device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.dp_policy.conditional_sample(
            cond_data, cond_mask, global_cond=global_cond
        )
        naction_pred = nsample[..., :Da]
        action_pred  = self.dp_policy.normalizer["action"].unnormalize(naction_pred)

        To    = self.dp_policy.n_obs_steps
        start = To
        end   = start + self.dp_policy.n_action_steps
        action = action_pred[:, start:end]

        return {"action": action, "action_pred": action_pred}

    def predict_beta(self, obs: torch.Tensor, option_id: torch.Tensor) -> torch.Tensor:
        """Return β probability (sigmoid) for the given obs/option pair."""
        B     = obs.shape[0]
        nobs  = self.dp_policy.normalizer["obs"].normalize(obs)
        nobs_flat = nobs[:, :self.dp_policy.n_obs_steps].reshape(B, -1)
        opt_emb   = self.option_embed(option_id)
        beta_in   = torch.cat([nobs_flat, opt_emb], dim=-1)
        return torch.sigmoid(self.termination_head(beta_in).squeeze(-1))

    def reset(self):
        pass

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def compute_loss(self, batch: dict) -> dict[str, torch.Tensor]:
        """
        batch keys: obs (B,T,D), action (B,H,Da), option_id (B,), beta_label (B,)
        Returns dict with loss_diffusion, loss_termination.
        """
        obs       = batch["obs"]["obs"].to(self.device)    # (B, T_obs, obs_dim)
        action    = batch["action"].to(self.device)        # (B, H, Da)
        option_id = batch["option_id"].to(self.device)     # (B,)
        beta_lbl  = batch["beta_label"].to(self.device)    # (B,)

        B = obs.shape[0]

        # ── Normalise ────────────────────────────────────────────────────────
        nobs    = self.dp_policy.normalizer["obs"].normalize(obs)
        naction = self.dp_policy.normalizer["action"].normalize(action)

        # ── Global cond = obs_flat + option_embed ────────────────────────────
        nobs_flat   = nobs[:, :self.dp_policy.n_obs_steps].reshape(B, -1)
        opt_emb     = self.option_embed(option_id)
        global_cond = torch.cat([nobs_flat, opt_emb], dim=-1)

        # ── Diffusion loss ───────────────────────────────────────────────────
        trajectory  = naction
        noise       = torch.randn_like(trajectory)
        timesteps   = torch.randint(
            0, self.dp_policy.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        noisy_traj  = self.dp_policy.noise_scheduler.add_noise(trajectory, noise, timesteps)
        pred        = self.dp_policy.model(noisy_traj, timesteps, global_cond=global_cond)

        if self.dp_policy.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.dp_policy.noise_scheduler.config.prediction_type == "sample":
            target = trajectory
        else:
            raise ValueError(self.dp_policy.noise_scheduler.config.prediction_type)

        loss_diffusion = nn.functional.mse_loss(pred, target)

        # ── Termination loss ─────────────────────────────────────────────────
        beta_in    = torch.cat([nobs_flat.detach(), opt_emb.detach()], dim=-1)
        beta_logit = self.termination_head(beta_in).squeeze(-1)
        loss_term  = nn.functional.binary_cross_entropy_with_logits(
            beta_logit, beta_lbl
        )

        return {
            "loss_diffusion":   loss_diffusion,
            "loss_termination": loss_term,
        }
