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
    termination_stop_grad: bool = True
    termination_label_smoothing: float = 0.0
    imagenet_init: bool = False
    escape_relabeling: bool = False  # True when dataset uses positive_negative expand-all


# ---------------------------------------------------------------------------
# Low policy (subclass DP)
# ---------------------------------------------------------------------------

TerminationInput = str  # "bottleneck" | "obs" | "both" | "bottleneck_ddim5"

_TERMINATION_INPUTS = ("bottleneck", "obs", "both", "bottleneck_ddim5")


def normalize_termination_input(value: str) -> str:
    """Parse ``low_policy.termination_input``."""
    mode = str(value).strip().lower()
    if mode not in _TERMINATION_INPUTS:
        raise ValueError(
            f"termination_input must be one of {_TERMINATION_INPUTS}, got {value!r}"
        )
    return mode


def obs_termination_input_dim(
    obs_feature_dim: int,
    n_obs_steps: int,
    option_embed_dim: int,
) -> int:
    """β MLP input size for ``termination_input=obs``: obs_feat * n_obs_steps + option_embed."""
    return int(obs_feature_dim) * int(n_obs_steps) + int(option_embed_dim)


def both_termination_input_dim(
    obs_feature_dim: int,
    n_obs_steps: int,
    option_embed_dim: int,
    bottleneck_dim: int,
) -> int:
    """β MLP input size for ``termination_input=both``: obs+option concat + bottleneck."""
    return obs_termination_input_dim(obs_feature_dim, n_obs_steps, option_embed_dim) + int(bottleneck_dim)


def _load_imagenet_pretrained_conv_weights(obs_encoder: "nn.Module") -> None:
    """Copy ImageNet pretrained conv weights into all ResNet18Conv backbones in obs_encoder.

    The robomimic path initializes ResNet18Conv with pretrained=False. This function
    loads torchvision IMAGENET1K_V1 weights and copies all shape-compatible parameters
    (conv kernels + norm weight/bias) into each backbone via strict=False, skipping
    BN running stats that don't exist in GroupNorm.
    """
    import torchvision.models as tvm
    from robomimic.models.base_nets import ResNet18Conv

    pretrained = tvm.resnet18(weights="IMAGENET1K_V1")
    pretrained_sd = torch.nn.Sequential(*(list(pretrained.children())[:-2])).state_dict()

    loaded = 0
    for module in obs_encoder.modules():
        if isinstance(module, ResNet18Conv):
            module.nets.load_state_dict(pretrained_sd, strict=False)
            loaded += 1

    if loaded == 0:
        raise RuntimeError("imagenet_init=True but no ResNet18Conv found in obs_encoder")
    print(f"[LowPolicy] ImageNet pretrained conv weights loaded into {loaded} ResNet18Conv backbone(s)")


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
            if cfg.imagenet_init:
                _load_imagenet_pretrained_conv_weights(self.obs_encoder)
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

            if self.termination_input in ("bottleneck", "bottleneck_ddim5"):
                # Both use the U-Net bottleneck vector as β MLP input.
                # bottleneck: expert actions at t=0 (cheap, deterministic).
                # bottleneck_ddim5: DDIM-5 generated actions at t=0 (closer to inference).
                if cfg.termination_head.bottleneck_dim != expected_bottleneck:
                    raise ValueError(
                        "termination_head.bottleneck_dim="
                        f"{cfg.termination_head.bottleneck_dim} must match U-Net mid dim "
                        f"{expected_bottleneck} when termination_input={self.termination_input}"
                    )
                th_cfg = cfg.termination_head
            elif self.termination_input == "both":
                # Concat [obs+option features, bottleneck] → β MLP.
                th_cfg = TerminationHeadConfig(
                    bottleneck_dim=both_termination_input_dim(
                        int(self.obs_feature_dim),
                        int(self.n_obs_steps),
                        int(cfg.option_embed_dim),
                        expected_bottleneck,
                    ),
                    hidden_dim=cfg.termination_head.hidden_dim,
                    num_layers=cfg.termination_head.num_layers,
                    use_chunk_cursor=cfg.termination_head.use_chunk_cursor,
                )
            else:  # obs
                th_cfg = TerminationHeadConfig(
                    bottleneck_dim=obs_termination_input_dim(
                        int(self.obs_feature_dim),
                        int(self.n_obs_steps),
                        int(cfg.option_embed_dim),
                    ),
                    hidden_dim=cfg.termination_head.hidden_dim,
                    num_layers=cfg.termination_head.num_layers,
                    use_chunk_cursor=cfg.termination_head.use_chunk_cursor,
                )
            self.termination_head = TerminationHead(th_cfg)

            # DDIM scheduler for bottleneck_ddim5: 5-step sampling during training.
            # self.noise_scheduler is set by parent __init__ before _init_soda_modules.
            if self.termination_input == "bottleneck_ddim5":
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler as _DDIMScheduler
                ns = self.noise_scheduler
                self._ddim_scheduler = _DDIMScheduler(
                    num_train_timesteps=ns.config.num_train_timesteps,
                    beta_start=ns.config.beta_start,
                    beta_end=ns.config.beta_end,
                    beta_schedule=ns.config.beta_schedule,
                    clip_sample=ns.config.clip_sample,
                    prediction_type=ns.config.prediction_type,
                )
                self._ddim_scheduler.set_timesteps(5)

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

        def predict_termination_logit(
            self,
            features: torch.Tensor,
            chunk_cursor: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """β MLP on bottleneck or obs+ω features (optional stop-grad on input)."""
            return self.termination_head(
                features, stop_grad=self.cfg.termination_stop_grad, chunk_cursor=chunk_cursor
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

            # All remaining modes need global_cond (obs + option embedding).
            if trajectory is None or global_cond is None:
                nactions = self.normalizer["action"].normalize(batch["action"])
                trajectory = nactions
                global_cond = self.build_global_cond(
                    self.encode_obs_global_features(nobs, batch_size),
                    option_id,
                )

            t_zero = torch.zeros(batch_size, device=device, dtype=torch.long)

            if self.termination_input == "bottleneck":
                # Expert actions at t=0 → bottleneck (cheap, deterministic).
                with grad_ctx:
                    _, features = self.forward_unet_with_bottleneck(
                        trajectory, t_zero, global_cond
                    )
                return features

            if self.termination_input == "bottleneck_ddim5":
                # DDIM-5 generated actions at t=0 → bottleneck (closer to inference).
                # Sampling always no_grad (can't backprop through 5 DDIM steps anyway).
                x_hat = self._sample_ddim5(global_cond)  # no_grad inside
                with grad_ctx:
                    _, features = self.forward_unet_with_bottleneck(
                        x_hat, t_zero, global_cond
                    )
                return features

            # both: concat [obs+option features, bottleneck from expert actions at t=0]
            with grad_ctx:
                obs_cond = global_cond  # already [obs_feat || option_embed]
                _, bottleneck = self.forward_unet_with_bottleneck(
                    trajectory, t_zero, global_cond
                )
            return torch.cat([obs_cond, bottleneck], dim=-1)

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

        def _sample_ddim5(self, global_cond: torch.Tensor) -> torch.Tensor:
            """Generate action chunk via DDIM in 5 steps (no gradient).

            Used by ``termination_input=bottleneck_ddim5`` during training to produce
            a policy-generated plan (closer to inference distribution than expert actions).
            Returns normalized actions of shape ``(B, horizon, action_dim)``; caller
            feeds this to ``forward_unet_with_bottleneck`` at t=0 for the bottleneck.
            """
            B = global_cond.shape[0]
            device = global_cond.device
            dtype = global_cond.dtype
            motion_dim = self.action_dim - 1  # exclude duration channel
            x = torch.randn(B, self.horizon, motion_dim, device=device, dtype=dtype)
            with torch.no_grad():
                for t in self._ddim_scheduler.timesteps:
                    t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
                    # Pad to action_dim with zeros for duration channel (not denoised)
                    x_input = torch.cat([
                        x,
                        torch.zeros(B, self.horizon, 1, device=device, dtype=dtype),
                    ], dim=-1)
                    noise_pred = self.model(x_input, t_batch, global_cond=global_cond)
                    # Only step the motion dims
                    x = self._ddim_scheduler.step(
                        noise_pred[..., :motion_dim], t, x
                    ).prev_sample
            # Return as full action_dim with zero duration channel (only motion dims used for bottleneck)
            return torch.cat([
                x,
                torch.zeros(B, self.horizon, 1, device=device, dtype=dtype),
            ], dim=-1)

        def beta_logits_from_batch(self, batch: dict[str, Any]) -> torch.Tensor:
            """Termination logits for val metrics (matches ``compute_loss`` β branch)."""
            features = self.termination_features_from_batch(batch)
            chunk_cursor = batch.get("chunk_cursor", None)
            return self.predict_termination_logit(features, chunk_cursor=chunk_cursor)

        @torch.no_grad()
        def predict_beta(
            self,
            obs_dict: dict[str, Any],
            option_id: torch.Tensor,
            action_plan: torch.Tensor | None = None,
            *,
            diffusion_t: int = 0,
            chunk_cursor: float | torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            Inference β from ``termination_input`` features.

            ``bottleneck``: U-Net @ ``(cached plan, t=diffusion_t, obs, ω)``.
            ``obs``: ``global_cond(obs, ω)`` only (``action_plan`` ignored).

            ``chunk_cursor``: scalar or ``(B,)`` tensor ∈ [0, 1] indicating progress
            through the current option segment. Passed to the termination head when
            ``use_chunk_cursor=True``.
            """
            nobs = self.normalizer.normalize(obs_dict)
            value = next(iter(nobs.values()))
            batch_size = int(value.shape[0])
            device = value.device
            dtype = value.dtype
            option_id = option_id.reshape(-1).long().to(device)

            cursor_t: torch.Tensor | None = None
            if chunk_cursor is not None:
                if torch.is_tensor(chunk_cursor):
                    cursor_t = chunk_cursor.reshape(batch_size).to(device=device, dtype=dtype)
                else:
                    cursor_t = torch.full((batch_size,), float(chunk_cursor), device=device, dtype=dtype)

            if self.termination_input == "obs":
                obs_feat = self.encode_obs_global_features(nobs, batch_size)
                features = self.build_global_cond(obs_feat, option_id)
                return torch.sigmoid(self.predict_termination_logit(features, chunk_cursor=cursor_t))

            # All bottleneck-based modes need global_cond and a trajectory.
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
            _, bottleneck = self.forward_unet_with_bottleneck(traj, t, global_cond)

            if self.termination_input in ("bottleneck", "bottleneck_ddim5"):
                # At inference, both use the cached DDPM plan (action_plan).
                # bottleneck_ddim5's DDIM sampling is training-only; inference path is identical.
                return torch.sigmoid(self.predict_termination_logit(bottleneck, chunk_cursor=cursor_t))

            # both: concat [obs+option, bottleneck]
            features = torch.cat([global_cond, bottleneck], dim=-1)
            return torch.sigmoid(self.predict_termination_logit(features, chunk_cursor=cursor_t))

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
            mask_diffusion_on_positive: bool = False,
            skip_diffusion: bool = False,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            """
            π_low training loss: diffusion MSE + weighted termination BCE.

            Returns ``(loss_total, log_dict)`` where ``log_dict`` has three scalars for
            logging (``loss``, ``loss_diffusion``, ``loss_termination``). Only
            ``loss_total`` is backpropped.

            When ``skip_diffusion=True`` the noisy U-Net forward is skipped and
            ``loss_diff=0``. Use in phase 2 where backbone lr=0 to save compute.
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
            chunk_cursor = batch["chunk_cursor"].reshape(-1).to(nactions.device) if "chunk_cursor" in batch else None
            # is_escape: True for synthetic wrong-option examples (positive_negative training).
            # These must not train the diffusion head — the wrong option_id in global_cond
            # would corrupt U-Net if paired with real actions in the diffusion loss.
            is_escape = batch.get("is_escape")
            if is_escape is not None:
                is_escape = is_escape.reshape(-1).bool().to(nactions.device)

            # Option-conditioned global cond (obs encoder + embed(ω)).
            global_cond = self.build_global_cond(
                self.encode_obs_global_features(nobs, batch_size),
                option_id,
            )
            trajectory = nactions

            # --- Diffusion loss ---
            if skip_diffusion:
                loss_diff = nactions.new_zeros(())
            else:
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

                neg_mask = (beta_label == 0).float() if mask_diffusion_on_positive else None
                # Escape examples must not contribute to diffusion loss regardless of
                # mask_diffusion_on_positive — their option_id is wrong for these actions.
                if is_escape is not None and is_escape.any():
                    non_escape = (~is_escape).float()
                    neg_mask = (neg_mask * non_escape) if neg_mask is not None else non_escape

                if class_weights is not None:
                    per_sample_diff = diffusion_loss_per_sample(pred, target, loss_mask)
                    if neg_mask is not None:
                        per_sample_diff = per_sample_diff * neg_mask
                    loss_diff = weighted_option_mean(per_sample_diff, option_id, class_weights)
                else:
                    if neg_mask is not None:
                        per_sample_diff = diffusion_loss_per_sample(pred, target, loss_mask)
                        n_neg = neg_mask.sum().clamp(min=1.0)
                        loss_diff = (per_sample_diff * neg_mask).sum() / n_neg
                    else:
                        loss_diff = diffusion_loss(pred, target, loss_mask)

            # --- Termination loss (``no_grad`` inside features; β MLP receives grads) ---
            term_features = self.termination_features_from_batch(
                batch,
                trajectory=trajectory,
                global_cond=global_cond,
            )
            beta_logit = self.predict_termination_logit(term_features, chunk_cursor=chunk_cursor)
            pos_weight = self.cfg.termination_pos_weight
            label_smoothing = float(self.cfg.termination_label_smoothing)

            if class_weights is not None:
                loss_term = weighted_option_mean(
                    termination_bce_loss_per_sample(
                        beta_logit, beta_label, pos_weight=pos_weight, label_smoothing=label_smoothing
                    ),
                    option_id,
                    class_weights,
                )
            else:
                loss_term = termination_bce_loss(
                    beta_logit, beta_label, pos_weight=pos_weight, label_smoothing=label_smoothing
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
