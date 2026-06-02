"""
Hierarchical SODA controller for sim eval (π_high + π_low + termination β).

Mirrors the DP ``BaseImagePolicy.predict_action`` interface. π_high selects ω;
``LowLevelChunkExecutor`` runs π_low until β fires (segment exit). If the
decompressed native chunk is exhausted before β fires, π_low replans under the
same ω — execution is not capped at ``n_action_steps``.

Each sim step:

1. **β (every step, including after replan):** ``pi_low.predict_beta`` on cached plan.
2. **Segment exit** when β fires → π_high resamples ω, clear cache, replan.
3. **Replan diffusion** when cache is empty or native chunk exhausted (same ω).
4. **Return** the next row from ``action_unstretched``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from soda.inference.low_level_executor import (
    HierarchicalPolicyConfig,
    LowLevelChunkExecutor,
    LowLevelStepResult,
)
from soda.option_discovery.supervised.pusht.frame_overlays import (
    LowPolicyOverlayInfo,
    duration_channel_stats,
)

__all__ = ["HierarchicalPolicy", "HierarchicalPolicyConfig"]

# Chunks this short (in native decoded steps) are treated as pathological —
# the same option is excluded from π_high's next prediction to break the loop.
_SHORT_CHUNK_STEPS = 2


class HierarchicalPolicy:
    """π_high selects ω; π_low diffuses chunks; β ends a segment."""

    def __init__(
        self,
        *,
        device: Any,
        config: HierarchicalPolicyConfig,
        checkpoint: Path | None = None,
        pi_high: Any | None = None,
        pi_low: Any | None = None,
        pi_low_weak: Any | None = None,
        external_beta: Any | None = None,
    ) -> None:
        self.device = device
        self.dtype = torch.float32
        self.config = config
        self.checkpoint = checkpoint
        self._pi_high = pi_high
        self._pi_low = pi_low
        self._pi_low_weak = pi_low_weak
        self._external_beta = external_beta
        self._executor: LowLevelChunkExecutor | None = None
        self._current_option_id: torch.Tensor | None = None
        self._prev_option_id: int | None = None  # None = episode start (null token)
        self._last_overlay: LowPolicyOverlayInfo = LowPolicyOverlayInfo()
        self._last_option_id: int | None = None
        self._last_fired_beta: torch.Tensor | None = None  # beta that triggered most recent option switch
        self._short_chunk_flag: bool = False  # set when last replan produced ≤ _SHORT_CHUNK_STEPS; forces option switch

    @property
    def last_overlay(self) -> LowPolicyOverlayInfo:
        return self._last_overlay

    @property
    def last_option_id(self) -> int | None:
        return self._last_option_id

    def set_control(
        self,
        *,
        beta_transition: float | None = None,
        beta_diffusion_t: int | None = None,
    ) -> None:
        """Override β eval knobs after construction."""
        updates: dict[str, Any] = {}
        if beta_transition is not None:
            updates["beta_transition"] = float(beta_transition)
        if beta_diffusion_t is not None:
            updates["beta_diffusion_t"] = int(beta_diffusion_t)
        if updates:
            self.config = replace(self.config, **updates)

    def _require_policies(self) -> tuple[Any, Any]:
        if self._pi_high is None or self._pi_low is None:
            raise NotImplementedError(
                "Attach pi_high and pi_low on HierarchicalPolicy before predict_action."
            )
        return self._pi_high, self._pi_low

    def _executor_for(self, pi_low: Any) -> LowLevelChunkExecutor:
        if self._executor is None or self._executor.pi_low is not pi_low:
            bid_sampler = None
            if self._pi_low_weak is not None:
                from soda.eval.bid_sampler import BIDSampler
                bid_sampler = BIDSampler(pi_low, self._pi_low_weak)
            self._executor = LowLevelChunkExecutor(
                pi_low, self.config,
                external_beta_fn=self._external_beta,
                bid_sampler=bid_sampler,
            )
        else:
            self._executor.config = self.config
        return self._executor

    def reset(self) -> None:
        self._current_option_id = None
        self._prev_option_id = None
        self._last_overlay = LowPolicyOverlayInfo()
        self._last_option_id = None
        self._last_fired_beta = None
        self._short_chunk_flag = False
        if self._executor is not None:
            self._executor.reset()
        elif self._pi_low is not None and hasattr(self._pi_low, "reset"):
            self._pi_low.reset()
        if self._pi_high is not None and hasattr(self._pi_high, "reset"):
            self._pi_high.reset()

    def _overlay_from_result(self, result: LowLevelStepResult) -> LowPolicyOverlayInfo:
        beta_val = None
        # If a switch just happened, show the beta that fired rather than the new plan's beta.
        fired = self._last_fired_beta
        if fired is not None:
            beta_val = float(fired.reshape(-1)[0].detach().cpu().item())
        elif result.beta is not None:
            beta_val = float(result.beta.reshape(-1)[0].detach().cpu().item())
        dur_mean = dur_std = decoded_native = horizon = None
        pred = result.action_pred
        if pred is None and self._executor is not None:
            pred = self._executor.cached_action_pred
        if pred is not None and self._pi_low is not None:
            pred_np = pred.detach().cpu().numpy()
            try:
                hor = int(self._pi_low.horizon)
                dur_mean, dur_std, decoded_native = duration_channel_stats(
                    pred_np, horizon=hor
                )
                horizon = hor
            except Exception:
                pass
        cursor = self._executor.chunk_cursor if self._executor is not None else None
        return LowPolicyOverlayInfo(
            beta=beta_val,
            duration_mean=dur_mean,
            duration_std=dur_std,
            horizon=horizon,
            decoded_native_steps=decoded_native,
            chunk_cursor=cursor,
            replan=result.replanned,
            beta_threshold=float(self.config.beta_transition),
        )

    def _sample_option(
        self,
        pi_high: Any,
        obs_dict: dict[str, Any],
        *,
        exclude_option_id: int | None = None,
    ) -> torch.Tensor:
        return pi_high.sample_option(
            pi_high.encode_obs(obs_dict),
            prev_option_id=self._prev_option_id,
            exclude_option_id=exclude_option_id,
        )

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Return the next single env action from the hierarchical stack.

        ``soda_runner`` calls this once per sim step. Low-level segments end only
        when β fires; chunk exhaustion triggers replan under the current ω.
        """
        pi_high, pi_low = self._require_policies()
        executor = self._executor_for(pi_low)

        # open_loop: resample ω from π_high after every full native chunk; β disabled.
        # duration_termination: receding-horizon π_low (n_action_steps cap); π_high called
        #   only when the chunk is exhausted before n_action_steps (duration channel fires).
        #   Long chunks (≥ n_action_steps) replan π_low under the same ω.
        if self.config.open_loop:
            if self._current_option_id is None or executor.needs_replan:
                if self._current_option_id is not None:
                    self._prev_option_id = int(self._current_option_id.reshape(-1)[0].item())
                self._current_option_id = self._sample_option(pi_high, obs_dict)
            result = executor.step(obs_dict, self._current_option_id)
        elif self.config.duration_termination and self.config.high_monitors_every_step:
            # π_high monitors every step. Only clear cache and switch on option change.
            # Same-option prediction → keep executing; executor handles duration replan
            # internally.
            # If the last replan produced a very short chunk (≤ _SHORT_CHUNK_STEPS), force
            # π_high to pick a different option — same-option re-selection would immediately
            # produce another short chunk and loop forever.
            curr_id = (
                int(self._current_option_id.reshape(-1)[0].item())
                if self._current_option_id is not None
                else -1
            )
            exclude_id = (
                curr_id
                if self._short_chunk_flag and self._current_option_id is not None
                else None
            )
            candidate = self._sample_option(pi_high, obs_dict, exclude_option_id=exclude_id)
            cand_id = int(candidate.reshape(-1)[0].item())
            if self._current_option_id is None or cand_id != curr_id:
                if self._current_option_id is not None:
                    self._prev_option_id = curr_id
                self._current_option_id = candidate
                executor.clear_cache()
                self._short_chunk_flag = False
            result = executor.step(obs_dict, self._current_option_id)
            if result.replanned:
                chunk_len = executor.last_replanned_chunk_len
                self._short_chunk_flag = (
                    chunk_len is not None and chunk_len <= _SHORT_CHUNK_STEPS
                )
        elif self.config.duration_termination:
            need_new_option = (
                self._current_option_id is None
                or (executor.needs_replan and executor.option_done)
            )
            if need_new_option:
                exclude_id: int | None = None
                if self._current_option_id is not None:
                    exclude_id = int(self._current_option_id.reshape(-1)[0].item())
                    self._prev_option_id = exclude_id
                self._current_option_id = self._sample_option(
                    pi_high, obs_dict, exclude_option_id=exclude_id
                )
            result = executor.step(obs_dict, self._current_option_id)
        else:
            if self._current_option_id is None:
                self._current_option_id = self._sample_option(pi_high, obs_dict)

            self._last_fired_beta = None
            for _ in range(8):
                result = executor.step(obs_dict, self._current_option_id)
                if result.beta_fired:
                    self._last_fired_beta = result.beta
                    self._prev_option_id = int(self._current_option_id.reshape(-1)[0].item())
                    self._current_option_id = self._sample_option(pi_high, obs_dict)
                    executor.clear_cache()
                    continue
                break
            else:
                raise RuntimeError("β fired repeatedly after replan; check beta_transition")

        self._last_option_id = (
            int(self._current_option_id.reshape(-1)[0].item())
            if self._current_option_id is not None
            else None
        )
        self._last_overlay = self._overlay_from_result(result)

        result_dict: dict[str, Any] = {
            "action": result.action,
            "action_pred": result.action_pred,
        }
        if result.beta is not None:
            result_dict["beta"] = result.beta
        return result_dict
