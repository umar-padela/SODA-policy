"""
Low-level option-conditioned diffusion policy π_low(a | s, ω) + duration channel.

Extends Columbia Diffusion Policy (subclass in ``soda/`` only — project_plan §4.1)
with:
- **Option conditioning:** ``nn.Embedding(ω)`` concatenated to global cond (not raw int).
- **D+1 actions:** last channel = normalized segment duration target for ``TemporalStretcher``.
- **Termination:** ``TerminationHead(stop_grad(U-Net bottleneck))`` — BCE, no grad into U-Net.
  Training β: second U-Net forward at ``t=0`` on clean expert suffix (matches ``predict_beta``).
  Training β uses a **second** U-Net forward at ``t=0`` on clean expert actions (matches ``predict_beta``).

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
- ``soda/dataset/option_aware_dataset.py`` — ``OptionAwareDataset``, ``beta_label``
- ``soda/dataset/temporal_stretch.py`` — stretch targets; mean-pool duration decode at infer
- ``soda/models/termination_head.py``, ``soda/training/losses_low.py``
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from soda.models.termination_head import TerminationHead, TerminationHeadConfig
from soda.models.unet_bottleneck import forward_unet_with_bottleneck


def _require_diffusion_policy_imports() -> None:
    from soda.eval.policy_loaders import _ensure_diffusion_policy_path

    _ensure_diffusion_policy_path()


def require_obs_as_global_cond(obs_as_global_cond: bool) -> bool:
    """
    Validate root-yaml ``obs_as_global_cond`` for π_low.

    SODA only supports DP hybrid **global conditioning** (obs encoded → ``global_cond``).
    The inpainting alternative (``obs_as_global_cond=False``) is not implemented.
    """
    if not bool(obs_as_global_cond):
        raise ValueError(
            "SODA π_low requires obs_as_global_cond=True (DP hybrid global conditioning). "
            "obs-as-inpainting mode (False) is not supported."
        )
    return True


def decode_segment_steps(
    action_chunk: torch.Tensor,
    horizon: int,
) -> float | torch.Tensor:
    """Torch wrapper — logic in ``soda.dataset.temporal_stretch.decode_segment_steps``."""
    from soda.dataset.temporal_stretch import decode_segment_steps as decode_segment_steps_np

    if action_chunk.ndim == 2:
        steps = decode_segment_steps_np(
            action_chunk.detach().cpu().numpy(), horizon
        )
        return float(steps)
    if action_chunk.ndim == 3:
        steps = decode_segment_steps_np(
            action_chunk.detach().cpu().numpy(), horizon
        )
        return torch.from_numpy(np.asarray(steps)).to(
            device=action_chunk.device, dtype=torch.long
        )
    raise ValueError(
        f"action_chunk must be (T, D+1) or (B, T, D+1), got shape {tuple(action_chunk.shape)}"
    )


def _patch_shape_meta_duration_channel(shape_meta: dict) -> dict:
    """Return a copy of ``shape_meta`` with action dim increased by 1 (duration channel)."""
    meta = copy.deepcopy(shape_meta)
    action_shape = meta["action"]["shape"]
    if len(action_shape) != 1:
        raise ValueError(f"Expected 1-D action shape, got {action_shape}")
    meta["action"]["shape"] = (int(action_shape[0]) + 1,)
    return meta


@contextmanager
def _widen_unet_global_cond(option_embed_dim: int):
    """Widen U-Net ``global_cond_dim`` by ``option_embed_dim`` during parent ``__init__``."""
    import diffusion_policy.policy.diffusion_unet_hybrid_image_policy as dp_policy_mod

    original_unet = dp_policy_mod.ConditionalUnet1D
    extra = int(option_embed_dim)

    class _WidenGlobalCondUnet(original_unet):
        def __init__(self, *args, global_cond_dim=None, **kwargs):
            if global_cond_dim is not None:
                global_cond_dim = int(global_cond_dim) + extra
            super().__init__(*args, global_cond_dim=global_cond_dim, **kwargs)

    dp_policy_mod.ConditionalUnet1D = _WidenGlobalCondUnet
    try:
        yield
    finally:
        dp_policy_mod.ConditionalUnet1D = original_unet


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LowPolicyConfig:
    """
    SODA-specific hyperparameters (Hydra ``low_policy:`` block).

    DP backbone fields (``horizon``, ``n_obs_steps``, U-Net dims, noise scheduler, …)
    stay in the upstream Hydra config / checkpoint — mirror ``dp_frozen.yaml``
    and extend ``shape_meta`` action dim to **D+1**.

    All fields here required — no silent defaults (``configs/pusht/soda_*.yaml``).
    """

    num_options: int
    option_embed_dim: int
    # Must match U-Net mid dim after upstream exposes bottleneck (infer from DP unet or cfg).
    bottleneck_dim: int
    termination_head: TerminationHeadConfig
    termination_loss_weight: float
    termination_pos_weight: float = 1.0
    termination_input: str = "bottleneck"
    # If True (default §8): detach β features before MLP; feature path under no_grad.
    # If False: L_termination backprops into termination_input backbone (ablation).
    termination_stop_grad: bool = True


# ---------------------------------------------------------------------------
# Low policy (subclass DP)
# ---------------------------------------------------------------------------

TerminationInput = str  # "bottleneck" | "obs"


def normalize_termination_input(value: str) -> str:
    """Parse ``low_policy.termination_input`` (``bottleneck`` or ``obs``)."""
    mode = str(value).strip().lower()
    if mode not in ("bottleneck", "obs"):
        raise ValueError(
            f"termination_input must be 'bottleneck' or 'obs', got {value!r}"
        )
    return mode


def obs_termination_input_dim(
    obs_feature_dim: int,
    n_obs_steps: int,
    option_embed_dim: int,
) -> int:
    """β MLP input size for ``termination_input=obs`` (encoder feat + ``option_embed(ω)``)."""
    return int(obs_feature_dim) * int(n_obs_steps) + int(option_embed_dim)


def _build_low_policy_class() -> type:
    """Lazy subclass of Columbia ``DiffusionUnetHybridImagePolicy`` (needs submodule on path)."""
    _require_diffusion_policy_imports()
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    class LowPolicy(DiffusionUnetHybridImagePolicy):
        """DP hybrid policy + option embed, D+1 duration channel, termination head."""

        def __init__(
            self,
            cfg: LowPolicyConfig,
            shape_meta: dict,
            noise_scheduler: Any,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            *,
            unet_down_dims: tuple[int, ...] = (256, 512, 1024),
            unet_diffusion_step_embed_dim: int = 256,
            unet_kernel_size: int = 5,
            unet_n_groups: int = 8,
            unet_cond_predict_scale: bool = True,
            obs_as_global_cond: bool = True,
            **kwargs: Any,
        ) -> None:
            require_obs_as_global_cond(obs_as_global_cond)
            soda_shape_meta = _patch_shape_meta_duration_channel(shape_meta)
            unet_kwargs = dict(
                down_dims=unet_down_dims,
                diffusion_step_embed_dim=unet_diffusion_step_embed_dim,
                kernel_size=unet_kernel_size,
                n_groups=unet_n_groups,
                cond_predict_scale=unet_cond_predict_scale,
            )

            # SODA option conditioning concat's to global_cond — inpainting mode not supported.
            kwargs.pop("obs_as_global_cond", None)

            # Parent: obs_encoder, U-Net, mask, scheduler.
            with _widen_unet_global_cond(cfg.option_embed_dim):
                super().__init__(
                    shape_meta=soda_shape_meta,
                    noise_scheduler=noise_scheduler,
                    horizon=horizon,
                    n_action_steps=n_action_steps,
                    n_obs_steps=n_obs_steps,
                    down_dims=unet_down_dims,
                    diffusion_step_embed_dim=unet_diffusion_step_embed_dim,
                    kernel_size=unet_kernel_size,
                    n_groups=unet_n_groups,
                    cond_predict_scale=unet_cond_predict_scale,
                    obs_as_global_cond=obs_as_global_cond,
                    **kwargs,
                )

            self.cfg = cfg
            self._init_soda_modules(cfg, unet_kwargs)

        def _init_soda_modules(
            self, cfg: LowPolicyConfig, unet_kwargs: dict[str, Any]
        ) -> None:
            from soda.dataset.temporal_stretch import TemporalStretcher

            expected_bottleneck = int(unet_kwargs["down_dims"][-1])
            if cfg.bottleneck_dim != expected_bottleneck:
                raise ValueError(
                    f"cfg.bottleneck_dim={cfg.bottleneck_dim} must match U-Net mid dim "
                    f"down_dims[-1]={expected_bottleneck}"
                )

            self.base_global_cond_dim = int(self.obs_feature_dim) * int(self.n_obs_steps)
            self.option_global_cond_dim = (
                self.base_global_cond_dim + int(cfg.option_embed_dim)
            )

            self.option_embed = nn.Embedding(cfg.num_options, cfg.option_embed_dim)
            self.termination_input = normalize_termination_input(cfg.termination_input)
            self._stretcher = TemporalStretcher(horizon=self.horizon)
            if self.termination_input == "bottleneck":
                if cfg.termination_head.bottleneck_dim != expected_bottleneck:
                    raise ValueError(
                        "termination_head.bottleneck_dim="
                        f"{cfg.termination_head.bottleneck_dim} must match U-Net mid dim "
                        f"{expected_bottleneck} when termination_input=bottleneck"
                    )
                th_cfg = cfg.termination_head
            else:
                th_cfg = TerminationHeadConfig(
                    bottleneck_dim=obs_termination_input_dim(
                        int(self.obs_feature_dim),
                        int(self.n_obs_steps),
                        int(cfg.option_embed_dim),
                    ),
                    hidden_dim=cfg.termination_head.hidden_dim,
                    num_layers=cfg.termination_head.num_layers,
                )
            self.termination_head = TerminationHead(th_cfg)

        # ------------------------------------------------------------------
        # Option conditioning
        # ------------------------------------------------------------------

        def option_embedding(self, option_id: torch.Tensor) -> torch.Tensor:
            """``option_id`` ``(B,)`` → ``(B, option_embed_dim)``."""
            option_id = option_id.reshape(-1).long()
            return self.option_embed(option_id)

        def build_global_cond(
            self,
            obs_features: torch.Tensor,
            option_id: torch.Tensor,
        ) -> torch.Tensor:
            """Concat obs global features with ``option_embed(ω)`` for U-Net conditioning."""
            return torch.cat([obs_features, self.option_embedding(option_id)], dim=-1)

        def predict_termination_logit(self, features: torch.Tensor) -> torch.Tensor:
            """β MLP on bottleneck or obs+ω features (optional stop-grad on input)."""
            return self.termination_head(
                features, stop_grad=self.cfg.termination_stop_grad
            )

        def termination_features_from_batch(
            self,
            batch: dict[str, Any],
            *,
            trajectory: torch.Tensor | None = None,
            global_cond: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            Features fed to ``TerminationHead`` for training / val metrics.

            With ``termination_stop_grad=True`` (default), feature extraction runs under
            ``no_grad`` and the MLP detaches its input (§8).

            With ``termination_stop_grad=False``, gradients from ``L_termination`` flow into
            the obs encoder and/or the U-Net clean-plan (t=0) forward.
            """
            from contextlib import nullcontext

            nobs = self.normalizer.normalize(batch["obs"])
            batch_size = int(batch["option_id"].reshape(-1).shape[0])
            option_id = batch["option_id"].reshape(-1).long()
            device = option_id.device
            grad_ctx = (
                torch.no_grad()
                if self.cfg.termination_stop_grad
                else nullcontext()
            )

            if self.termination_input == "obs":
                with grad_ctx:
                    obs_feat = self.encode_obs_global_features(nobs, batch_size)
                    return self.build_global_cond(obs_feat, option_id)

            if trajectory is None or global_cond is None:
                nactions = self.normalizer["action"].normalize(batch["action"])
                trajectory = nactions
                global_cond = self.build_global_cond(
                    self.encode_obs_global_features(nobs, batch_size),
                    option_id,
                )

            with grad_ctx:
                t_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
                _, features = self.forward_unet_with_bottleneck(
                    trajectory, t_zero, global_cond
                )
            return features

        def encode_obs_global_features(
            self, nobs: dict[str, torch.Tensor], batch_size: int
        ) -> torch.Tensor:
            """DP hybrid path: ``nobs`` → flattened ``(B, obs_feature_dim * n_obs_steps)``."""
            this_nobs = dict_apply(
                nobs,
                lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
            )
            nobs_features = self.obs_encoder(this_nobs)
            return nobs_features.reshape(batch_size, -1)

        def forward_unet_with_bottleneck(
            self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            global_cond: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """One U-Net forward; ``(pred, pooled_bottleneck (B, C))``."""
            return forward_unet_with_bottleneck(
                self.model, sample, timestep, global_cond, local_cond=None
            )

        def beta_logits_from_batch(self, batch: dict[str, Any]) -> torch.Tensor:
            """Termination logits for val metrics (matches ``compute_loss`` β branch)."""
            features = self.termination_features_from_batch(batch)
            return self.predict_termination_logit(features)

        @torch.no_grad()
        def predict_beta(
            self,
            obs_dict: dict[str, Any],
            option_id: torch.Tensor,
            action_plan: torch.Tensor | None = None,
            *,
            diffusion_t: int = 0,
        ) -> torch.Tensor:
            """
            Inference β from ``termination_input`` features.

            ``bottleneck``: U-Net @ ``(cached plan, t=diffusion_t, obs, ω)``.
            ``obs``: ``global_cond(obs, ω)`` only (``action_plan`` ignored).
            """
            nobs = self.normalizer.normalize(obs_dict)
            value = next(iter(nobs.values()))
            batch_size = int(value.shape[0])
            device = value.device
            dtype = value.dtype
            option_id = option_id.reshape(-1).long().to(device)

            if self.termination_input == "obs":
                obs_feat = self.encode_obs_global_features(nobs, batch_size)
                features = self.build_global_cond(obs_feat, option_id)
                return torch.sigmoid(self.predict_termination_logit(features))

            global_cond = self.build_global_cond(
                self.encode_obs_global_features(nobs, batch_size),
                option_id,
            )

            if action_plan is None:
                traj = torch.zeros(
                    batch_size, self.horizon, self.action_dim, device=device, dtype=dtype
                )
            else:
                traj = action_plan.to(device=device, dtype=dtype)
                if traj.ndim == 2:
                    traj = traj.unsqueeze(0)
                if traj.shape[0] != batch_size:
                    raise ValueError(
                        f"action_plan batch {traj.shape[0]} != obs batch {batch_size}"
                    )
                traj = self.normalizer["action"].normalize(traj)

            t = torch.full((batch_size,), int(diffusion_t), device=device, dtype=torch.long)
            _, features = self.forward_unet_with_bottleneck(traj, t, global_cond)
            return torch.sigmoid(self.predict_termination_logit(features))

        @torch.no_grad()
        def predict_action(
            self,
            obs_dict: dict[str, Any],
            option_id: torch.Tensor,
        ) -> dict[str, Any]:
            """Sample action chunk conditioned on ``option_id`` (DP path + option global cond)."""
            assert "past_action" not in obs_dict
            nobs = self.normalizer.normalize(obs_dict)
            value = next(iter(nobs.values()))
            batch_size = int(value.shape[0])
            device = self.device
            dtype = self.dtype
            t_horizon = self.horizon
            da = self.action_dim

            global_cond = self.build_global_cond(
                self.encode_obs_global_features(nobs, batch_size),
                option_id.reshape(-1).long(),
            )
            cond_data = torch.zeros((batch_size, t_horizon, da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            nsample = self.conditional_sample(
                cond_data,
                cond_mask,
                local_cond=None,
                global_cond=global_cond,
                **self.kwargs,
            )
            naction_pred = nsample[..., :da]
            action_pred = self.normalizer["action"].unnormalize(naction_pred)

            # Decode variable-length native actions from duration channel (project_plan §2).
            # Duration channel is the last dim; motion dims are [0, da-1).
            motion_dim = da - 1
            action_pred_np = action_pred.detach().cpu().numpy()  # (B, horizon, D+1)
            unstretched_list = []
            for i in range(batch_size):
                seg_steps = self._stretcher.decode_duration(action_pred_np[i])
                native = self._stretcher.unstretch(
                    action_pred_np[i, :, :motion_dim], seg_steps
                )  # (seg_steps, D)
                unstretched_list.append(native)
            # Pad to common length for batch tensors (B=1 in eval, no padding needed).
            max_native = max(a.shape[0] for a in unstretched_list)
            unstretched_np = np.zeros(
                (batch_size, max_native, motion_dim), dtype=np.float32
            )
            for i, arr in enumerate(unstretched_list):
                unstretched_np[i, : arr.shape[0]] = arr
            action_unstretched = torch.from_numpy(unstretched_np).to(
                device=device, dtype=dtype
            )  # (B, native_steps, D) — indexes from t=0 (anchor frame)

            return {
                "action": action_unstretched[:, : self.n_action_steps],
                "action_pred": action_pred,
                "action_unstretched": action_unstretched,
            }

        def compute_loss(
            self,
            batch: dict[str, Any],
            *,
            class_weights: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            """
            π_low training loss: diffusion MSE + weighted termination BCE.

            Returns ``(loss_total, log_dict)`` where ``log_dict`` has three scalars for
            logging (``loss``, ``loss_diffusion``, ``loss_termination``). Only
            ``loss_total`` is backpropped.

            ``L_termination`` → ``TerminationHead``; if ``termination_stop_grad`` (default),
            no grad into U-Net/encoder from β. If False, β also updates the t=0 U-Net path
            and/or obs encoder (ablation).

            ``termination_input=bottleneck``: second U-Net forward at ``t=0`` on clean expert
            plan (teacher-forced). ``termination_input=obs``: ``global_cond(obs, ω)`` only.
            Diffusion still trains at random ``t`` on the noisy trajectory.
            """
            from soda.training.losses_low import (
                diffusion_loss,
                diffusion_loss_per_sample,
                low_policy_total_loss,
                termination_bce_loss,
                termination_bce_loss_per_sample,
            )
            from soda.training.option_balance import weighted_option_mean

            assert "valid_mask" not in batch
            nobs = self.normalizer.normalize(batch["obs"])
            nactions = self.normalizer["action"].normalize(batch["action"])
            batch_size = int(nactions.shape[0])
            option_id = batch["option_id"].reshape(-1).long().to(nactions.device)
            beta_label = batch["beta_label"].reshape(-1).to(nactions.device)

            # Option-conditioned global cond (obs encoder + embed(ω)).
            global_cond = self.build_global_cond(
                self.encode_obs_global_features(nobs, batch_size),
                option_id,
            )

            # Forward diffusion at random t (same schedule as vanilla DP).
            trajectory = nactions
            condition_mask = self.mask_generator(trajectory.shape)
            noise = torch.randn_like(trajectory)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=trajectory.device,
            ).long()
            noisy_trajectory = self.noise_scheduler.add_noise(
                trajectory, noise, timesteps
            )
            loss_mask = ~condition_mask
            noisy_trajectory = noisy_trajectory.clone()
            noisy_trajectory[condition_mask] = trajectory[condition_mask]

            pred, _ = self.forward_unet_with_bottleneck(
                noisy_trajectory, timesteps, global_cond
            )

            pred_type = self.noise_scheduler.config.prediction_type
            if pred_type == "epsilon":
                target = noise
            elif pred_type == "sample":
                target = trajectory
            else:
                raise ValueError(f"Unsupported prediction type {pred_type}")

            # Termination features (``no_grad`` inside); β MLP *outside* so it receives grads.
            term_features = self.termination_features_from_batch(
                batch,
                trajectory=trajectory,
                global_cond=global_cond,
            )
            beta_logit = self.predict_termination_logit(term_features)
            pos_weight = self.cfg.termination_pos_weight

            if class_weights is not None:
                loss_diff = weighted_option_mean(
                    diffusion_loss_per_sample(pred, target, loss_mask),
                    option_id,
                    class_weights,
                )
                loss_term = weighted_option_mean(
                    termination_bce_loss_per_sample(
                        beta_logit, beta_label, pos_weight=pos_weight
                    ),
                    option_id,
                    class_weights,
                )
            else:
                loss_diff = diffusion_loss(pred, target, loss_mask)
                loss_term = termination_bce_loss(
                    beta_logit, beta_label, pos_weight=pos_weight
                )

            return low_policy_total_loss(
                loss_diff,
                loss_term,
                termination_weight=self.cfg.termination_loss_weight,
            )

        @staticmethod
        def decode_segment_steps(
            action_chunk: torch.Tensor,
            horizon: int,
        ) -> float | torch.Tensor:
            return decode_segment_steps(action_chunk, horizon)


    return LowPolicy


_low_policy_cls: type | None = None


def _get_low_policy_class() -> type:
    global _low_policy_cls
    if _low_policy_cls is None:
        _low_policy_cls = _build_low_policy_class()
    return _low_policy_cls


def __getattr__(name: str) -> type:
    if name == "LowPolicy":
        return _get_low_policy_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
