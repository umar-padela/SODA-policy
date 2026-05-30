"""
Fixed-ω π_low execution — ``LowLevelChunkExecutor`` without π_high.

Receding-horizon: replan after min(native_len, n_action_steps) steps.
When β fires, ``predict_action`` returns ``{"beta_fired": True}`` so the rollout
loop can exit — unlike HierarchicalPolicy which resamples ω and continues.
"""

from __future__ import annotations

from typing import Any

import torch

from soda.inference.hierarchical_controller import HierarchicalPolicyConfig
from soda.inference.low_level_executor import LowLevelChunkExecutor
from soda.option_discovery.supervised.pusht.frame_overlays import (
    LowPolicyOverlayInfo,
    duration_channel_stats,
)


class FixedOptionLowController:
    """
    π_low with fixed skill id ω (no π_high).

    Receding-horizon: replan after min(native_len, n_action_steps) steps (same as
    HierarchicalPolicy). When β fires, predict_action returns {"beta_fired": True}
    and the rollout loop exits — it does NOT replan and continue.
    """

    def __init__(
        self,
        pi_low: Any,
        fixed_option_id: int,
        config: HierarchicalPolicyConfig,
    ) -> None:
        self.pi_low = pi_low
        self.fixed_option_id = int(fixed_option_id)
        self.config = config
        self.device = self.pi_low.device
        self.dtype = torch.float32
        self._executor = LowLevelChunkExecutor(pi_low, config)
        self._option_id: torch.Tensor | None = None
        self._last_overlay = LowPolicyOverlayInfo()

    @property
    def last_overlay(self) -> LowPolicyOverlayInfo:
        return self._last_overlay

    def reset(self) -> None:
        self._option_id = None
        self._last_overlay = LowPolicyOverlayInfo()
        self._executor.reset()

    def _option_tensor(self, batch_size: int, device: Any) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            int(self.fixed_option_id),
            device=device,
            dtype=torch.long,
        )

    def _overlay_from_step(
        self,
        *,
        beta: torch.Tensor | None,
        replanned: bool,
        action_pred: torch.Tensor | None,
    ) -> LowPolicyOverlayInfo:
        beta_val = None
        if beta is not None:
            beta_val = float(beta.reshape(-1)[0].detach().cpu().item())
        dur_mean = dur_std = None
        decoded_native = None
        pred = action_pred if action_pred is not None else self._executor.cached_action_pred
        if pred is not None:
            pred_np = pred.detach().cpu().numpy()
            dur_mean, dur_std, decoded_native = duration_channel_stats(
                pred_np,
                horizon=int(self.pi_low.horizon),
            )
        cursor = self._executor.chunk_cursor
        horizon = int(self.pi_low.horizon)
        return LowPolicyOverlayInfo(
            beta=beta_val,
            duration_mean=dur_mean,
            duration_std=dur_std,
            horizon=horizon if dur_mean is not None else None,
            decoded_native_steps=decoded_native,
            chunk_cursor=cursor,
            replan=replanned,
            beta_threshold=float(self.config.beta_transition),
        )

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Return the next single env action (batch size 1).

        If β fires, returns ``{"beta_fired": True, "beta": <tensor>}`` with no
        action — the rollout loop must check this key and stop the episode.
        """
        value = next(iter(obs_dict.values()))
        batch_size = int(value.shape[0])
        if self._option_id is None:
            self._option_id = self._option_tensor(batch_size, value.device)

        result = self._executor.step(obs_dict, self._option_id)

        self._last_overlay = self._overlay_from_step(
            beta=result.beta,
            replanned=result.replanned,
            action_pred=result.action_pred,
        )

        if result.beta_fired:
            out: dict[str, Any] = {"beta_fired": True}
            if result.beta is not None:
                out["beta"] = result.beta
            return out

        result_dict: dict[str, Any] = {
            "action": result.action,
            "action_pred": result.action_pred,
        }
        if result.beta is not None:
            result_dict["beta"] = result.beta
        if result.action_unstretched is not None:
            result_dict["action_unstretched"] = result.action_unstretched
        return result_dict
