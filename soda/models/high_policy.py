"""
High-level option policy π_high(ω | s) via flow matching (project_plan §2, §7 row 15).

Trained on **segment-start** observations: given state at the beginning of each
option segment in expert demos, predict which discrete option ω was active.

Not used for frozen DP baseline eval; wired later via ``HierarchicalController``.

Structure (mirrors assignment1/hw1, different target + net)
-----------------------------------------------------------
HW1 ``FlowMatchingPolicy`` = temporal U-Net + ``FlowMatchingSchedule`` on action chunks.
π_high = ``ObsEncoder`` + ``OptionVelocityMLP`` + ``OptionFlowMatchingSchedule`` on
learned option embeddings (no temporal action axis).

References
----------
- ``project_plan.md`` §2 (π_high), §8B (FM design choices)
- ``soda/dataset/option_aware_dataset.py`` — ``build_option_segment_index``, segment.start
- ``assignment1/hw1/networks.py`` — ``FlowMatchingSchedule.interpolate``, ``.sample`` (Euler)
- ``assignment1/hw1/losses.py`` — ``flow_matching_loss`` training loop pattern
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HighPolicyConfig:
    """
    Hyperparameters for ``HighPolicy`` (populate from Hydra ``high_policy:`` block).

    All fields required — no silent defaults (set in ``configs/pusht/soda_*.yaml``).
    """

    num_options: int
    option_embed_dim: int
    global_feat_dim: int
    fm_hidden_dim: int
    fm_num_layers: int
    time_embed_dim: int
    num_inference_steps: int


# ---------------------------------------------------------------------------
# Flow matching schedule (assignment1 FlowMatchingSchedule analogue)
# ---------------------------------------------------------------------------


class OptionFlowMatchingSchedule:
    """
    Conditional OT flow matching in option embedding space ℝ^d.

    Same math as ``assignment1/hw1/networks.py::FlowMatchingSchedule``, but
    ``embed_dim`` replaces ``action_dim`` (no ``pred_horizon`` / temporal U-Net).

    Convention (match HW1 implementation):
        t=0 → noise x_0,  t=1 → data x_1 (option embedding).
        x_t = t * x_1 + (1 - t) * x_0
        target velocity v* = x_1 - x_0  (constant along the linear path)
    """

    def __init__(self, embed_dim: int, num_inference_steps: int = 10) -> None:
        self.embed_dim = int(embed_dim)
        self.num_inference_steps = int(num_inference_steps)

    @staticmethod
    def interpolate(
        x1: Any,
        t: Any,
    ) -> tuple[Any, Any]:
        """
        Build training pair (x_t, target_velocity).

        Parameters
        ----------
        x1
            Clean option targets ``(B, embed_dim)`` from ``nn.Embedding``.
        t
            Times in [0, 1], shape ``(B,)``.

        Returns
        -------
        x_t, velocity
            Both ``(B, embed_dim)``.

        -----------------------------------------------------------------
        - [ ] ``x0 = torch.randn_like(x1)``
        - [ ] ``t_reshaped = t.unsqueeze(-1)``
        - [ ] ``x_t = t_reshaped * x1 + (1 - t_reshaped) * x0``
        - [ ] ``velocity = x1 - x0``
        """
        # get random noise
        x0 = torch.randn_like(x1)
        # reshape t from (B,) to (B,1)
        t_reshaped = t.unsqueeze(-1)
        # interpolate between expert and noise
        xt = t_reshaped*x1 + (1-t_reshaped)*x0
        # compute velocity = difference between random noise and expert
        velocity = x1 - x0

        return xt, velocity

    def integrate_euler(
        self,
        predict_velocity: Callable[[Any, Any, Any], Any],
        global_feat: Any,
    ) -> Any:
        """
        Inference: integrate learned velocity field from noise to x_1 region.

        Parameters
        ----------
        predict_velocity
            Callable ``(x_t, global_feat, t) -> v_pred`` with ``t`` shape ``(B,)``.
        global_feat
            ``(B, global_feat_dim)`` from ``ObsEncoder``.

        Returns
        -------
        x_end
            ``(B, embed_dim)`` after ``num_inference_steps`` Euler steps.

        """
        B = global_feat.shape[0]
        device = global_feat.device

        x = torch.randn(B, self.embed_dim, device=device, dtype=global_feat.dtype)
        dt = 1.0 / self.num_inference_steps

        for i in range(self.num_inference_steps):
            t = torch.full((B,), i * dt, device=device, dtype=global_feat.dtype)
            velocity = predict_velocity(x, global_feat, t)
            x = x + dt * velocity

        return x


# ---------------------------------------------------------------------------
# Networks (HW1 uses TemporalNoisePredictor + ConditionalUnet1D instead)
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """
    Embed scalar flow time t ∈ [0, 1] for the velocity MLP.

    Same math as ``assignment1/hw1/networks.py::SinusoidalPosEmb``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """``(B,)`` or ``(B, 1)`` → ``(B, dim)``."""
        if t.ndim > 1:
            t = t.squeeze(-1)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ObsEncoder(nn.Module):
    """
    Map DP-style obs → ``global_feat`` for flow matching (frozen vision backbone).

    Wraps a pretrained ``obs_encoder`` + ``normalizer`` (Columbia hybrid DP by default).
    Forward matches ``DiffusionUnetHybridImagePolicy`` global-conditioning: all
    ``n_obs_steps`` frames are encoded and flattened.

    Weight source (pick one at construction)
    ----------------------------------------
    - ``ObsEncoder.from_dp_checkpoint(...)`` — frozen Columbia ``latest.ckpt``.
    - ``ObsEncoder.from_low_policy_checkpoint(...)`` — trained π_low vision + normalizer.

    Input ``obs`` keys: ``image`` ``(B, T, 3, H, W)``, ``agent_pos`` ``(B, T, 2)`` in [0, 1]
    float for images (same as ``OptionAwareDataset``).
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        normalizer: Any,
        n_obs_steps: int,
        *,
        global_feat_dim: int | None = None,
        source: str = "diffusion_policy",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.normalizer = normalizer
        self.n_obs_steps = int(n_obs_steps)
        self.source = source

        if freeze:
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if global_feat_dim is not None:
            self._global_feat_dim = int(global_feat_dim)
        else:
            self._global_feat_dim = self._infer_global_feat_dim()

    @property
    def global_feat_dim(self) -> int:
        """Feature size after ``forward`` (``obs_feature_dim * n_obs_steps``)."""
        return self._global_feat_dim

    def _infer_global_feat_dim(self) -> int:
        device = next(self.vision_encoder.parameters()).device
        dtype = next(self.vision_encoder.parameters()).dtype
        with torch.no_grad():
            dummy = {
                "image": torch.zeros(
                    1, self.n_obs_steps, 3, 96, 96, device=device, dtype=dtype
                ),
                "agent_pos": torch.zeros(
                    1, self.n_obs_steps, 2, device=device, dtype=dtype
                ),
            }
            return int(self.forward(dummy).shape[-1])

    @classmethod
    def from_dp_checkpoint(
        cls,
        checkpoint: str | Path,
        device: str = "cuda:0",
        *,
        freeze: bool = True,
    ) -> ObsEncoder:
        """Build encoder from frozen Columbia Push-T hybrid DP ``.ckpt``."""
        from soda.eval.policy_loaders import load_frozen_dp_obs_encoder

        vision_encoder, normalizer, n_obs_steps, feat_dim = load_frozen_dp_obs_encoder(
            checkpoint, device=device
        )
        return cls(
            vision_encoder,
            normalizer,
            n_obs_steps,
            global_feat_dim=feat_dim,
            source="diffusion_policy",
            freeze=freeze,
        )

    @classmethod
    def from_low_policy_checkpoint(
        cls,
        checkpoint: str | Path,
        device: str = "cuda:0",
        *,
        eval_cfg: Any | None = None,
        freeze: bool = True,
    ) -> ObsEncoder:
        """Build encoder from trained SODA π_low ``train_low`` checkpoint."""
        from soda.eval.policy_loaders import load_frozen_low_obs_encoder

        vision_encoder, normalizer, n_obs_steps, feat_dim = load_frozen_low_obs_encoder(
            checkpoint, device=device, eval_cfg=eval_cfg
        )
        return cls(
            vision_encoder,
            normalizer,
            n_obs_steps,
            global_feat_dim=feat_dim,
            source="soda_low",
            freeze=freeze,
        )

    @torch.no_grad()
    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns ``global_feat`` of shape ``(B, global_feat_dim)``.
        """
        nobs = self.normalizer.normalize(obs)
        B = int(nobs["image"].shape[0])
        To = self.n_obs_steps

        this_nobs = {
            key: tensor[:, :To, ...].reshape(-1, *tensor.shape[2:])
            for key, tensor in nobs.items()
        }
        feat = self.vision_encoder(this_nobs)
        return feat.reshape(B, -1)


class OptionVelocityMLP(nn.Module):
    """
    Velocity field v_θ(x_t, t | s) for option flow matching.

    HW1 analogue: ``TemporalNoisePredictor.forward(noisy_action, state, timestep)``.
    Here ``x_t`` replaces the flattened action chunk; ``global_feat`` replaces state.

    Conditioning matches ``ConditionalUnet1D``: sinusoidal ``t`` embed + small MLP, then
    concat with ``global_feat``; ``x_t`` is concatenated as the quantity being transported.
    A stack of ``num_layers`` hidden blocks (Mish activations) maps the joint input to
    ``v_pred``. Does not parameterize a Gaussian — velocity depends on ``x_t`` and ``t``.
    """

    def __init__(
        self,
        embed_dim: int,
        global_feat_dim: int,
        time_embed_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.global_feat_dim = int(global_feat_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        # Same pattern as assignment1/hw1/networks.py::ConditionalUnet1D (diffusion_step_encoder)
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(self.time_embed_dim * 4, self.time_embed_dim),
        )

        in_dim = self.embed_dim + self.global_feat_dim + self.time_embed_dim
        blocks: list[nn.Module] = [
            nn.Linear(in_dim, self.hidden_dim),
            nn.Mish(),
        ]
        for _ in range(self.num_layers - 1):
            blocks.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.Mish(),
                ]
            )
        blocks.append(nn.Linear(self.hidden_dim, self.embed_dim))
        self.mlp = nn.Sequential(*blocks)

    def forward(
        self,
        x_t: torch.Tensor,
        global_feat: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_t
            ``(B, option_embed_dim)``
        global_feat
            ``(B, global_feat_dim)``
        t
            ``(B,)`` in [0, 1]

        Returns
        -------
        v_pred
            ``(B, option_embed_dim)``
        """
        if t.ndim > 1:
            t = t.squeeze(-1)
        t = t.to(device=x_t.device, dtype=x_t.dtype)
        t_cond = self.time_encoder(t)
        h = torch.cat([x_t, global_feat, t_cond], dim=-1)
        return self.mlp(h)


# ---------------------------------------------------------------------------
# High-level policy (bundles schedule + encoder + velocity net + embedding table)
# ---------------------------------------------------------------------------


class VelocityFn(Protocol):
    """Callable signature shared by ``OptionVelocityMLP`` and ``integrate_euler``."""

    def __call__(self, x_t: Any, global_feat: Any, t: Any) -> Any: ...


class HighPolicy(nn.Module):
    """
    π_high(ω | s) via flow matching on learned option embeddings.

    Training API
    ------------
    - ``encode_obs(obs)`` → ``global_feat``
    - ``compute_loss(batch)`` → ``(loss_total, log_dict)``

    Inference API
    -------------
    - ``sample_option(global_feat)`` → ``(B,)`` int64 ω
    - ``forward(obs, option_id=None)`` dispatches train vs infer

    Construction
    ------------
    - ``HighPolicy(cfg, obs_encoder)`` — inject a built ``ObsEncoder``.
    - ``HighPolicy.from_dp_checkpoint(cfg, path, device=...)`` — frozen Columbia DP vision.
    - ``HighPolicy.from_low_policy_checkpoint(cfg, path, device=...)`` — frozen π_low vision.
    """

    def __init__(self, cfg: HighPolicyConfig, obs_encoder: ObsEncoder) -> None:
        super().__init__()
        self.cfg = cfg

        if obs_encoder.global_feat_dim != cfg.global_feat_dim:
            raise ValueError(
                f"obs_encoder.global_feat_dim={obs_encoder.global_feat_dim} "
                f"!= cfg.global_feat_dim={cfg.global_feat_dim}"
            )
        if cfg.time_embed_dim < 4 or cfg.time_embed_dim % 2 != 0:
            raise ValueError(
                f"time_embed_dim must be even and >= 4, got {cfg.time_embed_dim}"
            )

        self.schedule = OptionFlowMatchingSchedule(
            cfg.option_embed_dim, cfg.num_inference_steps
        )
        self.option_embed = nn.Embedding(cfg.num_options, cfg.option_embed_dim)
        self.obs_encoder = obs_encoder
        self.velocity_net = OptionVelocityMLP(
            embed_dim=cfg.option_embed_dim,
            global_feat_dim=cfg.global_feat_dim,
            time_embed_dim=cfg.time_embed_dim,
            hidden_dim=cfg.fm_hidden_dim,
            num_layers=cfg.fm_num_layers,
        )

    @classmethod
    def from_dp_checkpoint(
        cls,
        cfg: HighPolicyConfig,
        checkpoint: str | Path,
        device: str = "cpu",
    ) -> HighPolicy:
        """Build π_high with frozen Columbia hybrid DP ``ObsEncoder``."""
        obs_encoder = ObsEncoder.from_dp_checkpoint(checkpoint, device=device)
        return cls(cfg, obs_encoder).to(device)

    @classmethod
    def from_low_policy_checkpoint(
        cls,
        cfg: HighPolicyConfig,
        checkpoint: str | Path,
        device: str = "cpu",
        *,
        eval_cfg: Any | None = None,
    ) -> HighPolicy:
        """Build π_high with frozen vision from a trained π_low checkpoint."""
        obs_encoder = ObsEncoder.from_low_policy_checkpoint(
            checkpoint, device=device, eval_cfg=eval_cfg
        )
        return cls(cfg, obs_encoder).to(device)

    def option_target(self, option_id: torch.Tensor) -> torch.Tensor:
        """
        Ground-truth flow endpoint x_1 for training.

        Parameters
        ----------
        option_id
            ``(B,)`` long

        Returns
        -------
        x1
            ``(B, option_embed_dim)`` — ``option_embed(option_id)``
        """
        return self.option_embed(option_id)

    def encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """``obs`` → ``global_feat`` ``(B, global_feat_dim)``."""
        return self.obs_encoder(obs)

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        global_feat: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to ``OptionVelocityMLP`` (HW1: ``policy(xt, s_batch, t)``)."""
        return self.velocity_net(x_t, global_feat, t)

    def compute_loss(
        self, batch: dict[str, Any], *, class_weights: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        π_high training loss: flow-matching MSE on option embeddings.

        ``batch`` must contain ``obs`` and ``option_id`` (segment-start samples).
        """
        option_id = batch["option_id"].reshape(-1).long()
        global_feat = self.encode_obs(batch["obs"])
        loss = self._flow_matching_loss(
            global_feat, option_id, class_weights=class_weights
        )
        return loss, {"loss": float(loss.detach().item())}

    def _flow_matching_loss(
        self,
        global_feat: torch.Tensor,
        option_id: torch.Tensor,
        *,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x1 = self.option_target(option_id)
        batch_size = int(option_id.shape[0])
        t = torch.rand(batch_size, device=option_id.device, dtype=x1.dtype)
        x_t, v_tgt = self.schedule.interpolate(x1, t)
        v_pred = self.predict_velocity(x_t, global_feat, t)
        per_sample = F.mse_loss(v_pred, v_tgt, reduction="none").mean(dim=-1)
        if class_weights is not None:
            from soda.training.option_balance import weighted_option_mean

            return weighted_option_mean(per_sample, option_id, class_weights)
        return per_sample.mean()

    def discrete_decode_from_embedding(self, x_end: torch.Tensor) -> torch.Tensor:
        """
        Map continuous endpoint → discrete ω via nearest embedding row (L2).
        """
        # (B, D) vs (num_options, D) → (B, num_options)
        dists = torch.cdist(x_end, self.option_embed.weight, p=2)
        return dists.argmin(dim=-1)

    @torch.no_grad()
    def sample_option(self, global_feat: torch.Tensor) -> torch.Tensor:
        """Euler integration of the velocity field, then nearest-option decode."""
        was_training = self.training
        self.eval()
        try:
            x_end = self.schedule.integrate_euler(self.predict_velocity, global_feat)
            return self.discrete_decode_from_embedding(x_end)
        finally:
            if was_training:
                self.train()

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        option_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training: pass ``option_id`` → FM loss scalar.
        Inference: ``option_id is None`` → sampled ω ``(B,)``.
        """
        global_feat = self.encode_obs(obs)
        if option_id is None:
            return self.sample_option(global_feat)
        loss, _ = self.compute_loss({"obs": obs, "option_id": option_id})
        return loss
