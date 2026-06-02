"""
High-level option policy π_high(ω | s) — cross-entropy classifier.

Trained on **segment-start** observations: given state at the beginning of each
option segment in expert demos, predict which discrete option ω was active.

Architecture
------------
ObsEncoder (frozen ResNet from DP or π_low checkpoint) → global_feat
→ OptionClassifier MLP → logits (num_options,)
→ argmax → ω

Cross-entropy directly optimises option accuracy.  Flow matching was considered
but abandoned: for 3 discrete options a classifier is simpler, trains faster,
and argmax always returns a valid option (no averaging artefact).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HighPolicyConfig:
    """Hyperparameters for ``HighPolicy`` (populate from Hydra ``high_policy:`` block)."""

    num_options: int
    global_feat_dim: int
    hidden_dim: int       # classifier MLP hidden size
    num_layers: int       # classifier MLP depth (>= 1)
    dropout_rate: float = 0.0   # dropout after each hidden Mish; 0.0 = disabled
    label_smoothing: float = 0.0  # CE label smoothing; 0.0 = hard targets
    condition_on_prev_option: bool = False  # concatenate prev-option embedding to global_feat
    prev_option_embed_dim: int = 32         # embedding dim for prev-option; index num_options = null token


# ---------------------------------------------------------------------------
# Classifier MLP
# ---------------------------------------------------------------------------


class OptionClassifier(nn.Module):
    """
    MLP: global_feat → logits over num_options.

    Stack of ``num_layers`` Linear+Mish blocks followed by a final linear
    projection to ``num_options``.  No softmax here — callers use
    ``F.cross_entropy`` (training) or ``logits.argmax`` (inference).
    """

    def __init__(
        self,
        global_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_options: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        def _block(in_dim: int, out_dim: int) -> list[nn.Module]:
            mods: list[nn.Module] = [nn.Linear(in_dim, out_dim), nn.Mish()]
            if dropout_rate > 0.0:
                mods.append(nn.Dropout(dropout_rate))
            return mods

        blocks: list[nn.Module] = _block(global_feat_dim, hidden_dim)
        for _ in range(num_layers - 1):
            blocks += _block(hidden_dim, hidden_dim)
        blocks.append(nn.Linear(hidden_dim, num_options))
        self.net = nn.Sequential(*blocks)

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        """``(B, global_feat_dim)`` → ``(B, num_options)`` logits."""
        return self.net(global_feat)


# ---------------------------------------------------------------------------
# Observation encoder (unchanged — wraps frozen ResNet from DP / π_low ckpt)
# ---------------------------------------------------------------------------


class ObsEncoder(nn.Module):
    """
    Map DP-style obs → ``global_feat`` for the classifier.

    Wraps a pretrained ``obs_encoder`` + ``normalizer`` (Columbia hybrid DP).
    Forward matches ``DiffusionUnetHybridImagePolicy`` global-conditioning: all
    ``n_obs_steps`` frames are encoded and flattened.

    Weight source (pick one at construction)
    ----------------------------------------
    - ``ObsEncoder.from_dp_checkpoint(...)`` — frozen Columbia ``latest.ckpt``.
    - ``ObsEncoder.from_low_policy_checkpoint(...)`` — trained π_low vision + normalizer.
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
        else:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True

        if global_feat_dim is not None:
            self._global_feat_dim = int(global_feat_dim)
        else:
            self._global_feat_dim = self._infer_global_feat_dim()

    @property
    def global_feat_dim(self) -> int:
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
        """Build encoder from Columbia Push-T hybrid DP ``.ckpt``."""
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
        """Build encoder from trained SODA π_low checkpoint."""
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

    @classmethod
    def from_imagenet_init(
        cls,
        checkpoint: str | Path,
        device: str = "cpu",
        *,
        freeze: bool = True,
    ) -> ObsEncoder:
        """DP encoder architecture + GroupNorm, but ResNet weights from ImageNet only.

        Uses the DP checkpoint solely for normalizer stats and architecture spec.
        The vision backbone is NOT loaded from the Push-T fine-tuned weights — it
        starts from ImageNet, making π_high fully independent of any DP training run.
        """
        from soda.eval.policy_loaders import load_imagenet_dp_obs_encoder

        vision_encoder, normalizer, n_obs_steps, feat_dim = load_imagenet_dp_obs_encoder(
            checkpoint, device=device
        )
        return cls(
            vision_encoder,
            normalizer,
            n_obs_steps,
            global_feat_dim=feat_dim,
            source="imagenet_init",
            freeze=freeze,
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Returns ``global_feat`` of shape ``(B, global_feat_dim)``.

        No ``@torch.no_grad()`` so gradients can flow into the vision encoder
        when ``freeze=False``.  When frozen, PyTorch does not extend the
        computation graph into the encoder anyway.
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


# ---------------------------------------------------------------------------
# High-level policy
# ---------------------------------------------------------------------------


class HighPolicy(nn.Module):
    """
    π_high(ω | s) — cross-entropy classifier over discrete options.

    Training API
    ------------
    ``compute_loss(batch)`` → ``(loss, log_dict)``

    Inference API
    -------------
    ``sample_option(global_feat)`` → ``(B,)`` int64 ω  (argmax of logits)

    Construction
    ------------
    ``HighPolicy(cfg, obs_encoder)`` — inject a built ``ObsEncoder``.
    ``HighPolicy.from_dp_checkpoint(cfg, path)`` — frozen Columbia DP vision.
    ``HighPolicy.from_low_policy_checkpoint(cfg, path)`` — frozen π_low vision.
    """

    def __init__(self, cfg: HighPolicyConfig, obs_encoder: ObsEncoder) -> None:
        super().__init__()
        self.cfg = cfg

        if obs_encoder.global_feat_dim != cfg.global_feat_dim:
            raise ValueError(
                f"obs_encoder.global_feat_dim={obs_encoder.global_feat_dim} "
                f"!= cfg.global_feat_dim={cfg.global_feat_dim}"
            )

        self.obs_encoder = obs_encoder

        # prev-option embedding: index ``num_options`` is the null token (episode start)
        if cfg.condition_on_prev_option:
            self.prev_option_embedding: nn.Embedding | None = nn.Embedding(
                cfg.num_options + 1, cfg.prev_option_embed_dim
            )
        else:
            self.prev_option_embedding = None

        classifier_input_dim = cfg.global_feat_dim + (
            cfg.prev_option_embed_dim if cfg.condition_on_prev_option else 0
        )
        self.classifier = OptionClassifier(
            global_feat_dim=classifier_input_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_options=cfg.num_options,
            dropout_rate=cfg.dropout_rate,
        )

    @classmethod
    def from_dp_checkpoint(
        cls,
        cfg: HighPolicyConfig,
        checkpoint: str | Path,
        device: str = "cpu",
        *,
        freeze: bool = True,
    ) -> HighPolicy:
        """Build π_high with Columbia hybrid DP ``ObsEncoder``."""
        obs_encoder = ObsEncoder.from_dp_checkpoint(checkpoint, device=device, freeze=freeze)
        return cls(cfg, obs_encoder).to(device)

    @classmethod
    def from_low_policy_checkpoint(
        cls,
        cfg: HighPolicyConfig,
        checkpoint: str | Path,
        device: str = "cpu",
        *,
        eval_cfg: Any | None = None,
        freeze: bool = True,
    ) -> HighPolicy:
        """Build π_high with frozen vision from a trained π_low checkpoint."""
        obs_encoder = ObsEncoder.from_low_policy_checkpoint(
            checkpoint, device=device, eval_cfg=eval_cfg, freeze=freeze
        )
        return cls(cfg, obs_encoder).to(device)

    def encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """``obs`` → ``global_feat`` ``(B, global_feat_dim)``."""
        return self.obs_encoder(obs)

    def _build_classifier_input(
        self,
        global_feat: torch.Tensor,
        prev_option_id: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """Concatenate prev-option embedding to global_feat when conditioning is enabled."""
        if not self.cfg.condition_on_prev_option or self.prev_option_embedding is None:
            return global_feat
        B = global_feat.shape[0]
        if prev_option_id is None:
            prev_option_id = self.cfg.num_options  # null token
        if not isinstance(prev_option_id, torch.Tensor):
            prev_option_id = torch.full(
                (B,), prev_option_id, dtype=torch.long, device=global_feat.device
            )
        prev_emb = self.prev_option_embedding(prev_option_id.reshape(-1))
        return torch.cat([global_feat, prev_emb], dim=-1)

    def compute_loss(
        self,
        batch: dict[str, Any],
        *,
        class_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Cross-entropy loss on option logits.

        ``batch`` must contain ``obs`` and ``option_id`` (segment-start samples).
        ``class_weights`` — per-class weight tensor for imbalanced option counts.
        When ``condition_on_prev_option=True``, ``batch["prev_option_id"]`` is used.
        """
        option_id = batch["option_id"].reshape(-1).long()
        global_feat = self.encode_obs(batch["obs"])
        prev_option_id = batch.get("prev_option_id", None)
        classifier_input = self._build_classifier_input(global_feat, prev_option_id)
        logits = self.classifier(classifier_input)
        loss = F.cross_entropy(
            logits, option_id, weight=class_weights,
            label_smoothing=self.cfg.label_smoothing,
        )
        return loss, {"loss": float(loss.detach().item())}

    @torch.no_grad()
    def sample_option(
        self,
        global_feat: torch.Tensor,
        *,
        prev_option_id: torch.Tensor | int | None = None,
        exclude_option_id: int | None = None,
    ) -> torch.Tensor:
        """Argmax over classifier logits → ``(B,)`` int64 option id.

        When ``condition_on_prev_option=True``, pass ``prev_option_id`` (int or
        ``(B,)`` tensor).  ``None`` → null token (episode start).

        ``exclude_option_id`` — if set, mask that option's logit to -inf so the
        argmax selects the next-best option.  Useful to prevent the same option
        from being re-selected immediately after a duration-termination exit.
        """
        classifier_input = self._build_classifier_input(global_feat, prev_option_id)
        logits = self.classifier(classifier_input)
        if exclude_option_id is not None:
            logits = logits.clone()
            logits[:, exclude_option_id] = float("-inf")
        return logits.argmax(dim=-1)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        option_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training: pass ``option_id`` → CE loss scalar.
        Inference: ``option_id is None`` → sampled ω ``(B,)``.
        """
        global_feat = self.encode_obs(obs)
        if option_id is None:
            return self.sample_option(global_feat)
        loss, _ = self.compute_loss({"obs": obs, "option_id": option_id})
        return loss
