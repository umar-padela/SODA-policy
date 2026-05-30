"""
Shared π_low execute loop: one env action per step, β every step, receding-horizon replan.

Replan after ``min(native_len, n_action_steps)`` steps or when β fires.
What happens on β fire differs by caller:
  HierarchicalPolicy  → π_high resamples ω, executor replans.
  FixedOptionLowController → episode ends (β_fired propagated to rollout loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HierarchicalPolicyConfig:
    """Eval-time wiring (Hydra yaml + checkpoint loader)."""

    config_name: str = "soda_supervised"
    option_id_key: str = "option_id_supervised"
    env_action_dim: int = 2
    n_action_steps: int = 8  # receding-horizon window; replan after this many native steps
    beta_transition: float = 0.5
    beta_diffusion_t: int = 0
    open_loop: bool = False           # resample ω from π_high after every full native chunk (β ignored)
    duration_termination: bool = False  # exit to π_high when duration channel predicts < n_action_steps remain (β ignored)


@dataclass
class LowLevelStepResult:
    """One low-level sim step."""

    action: torch.Tensor
    action_pred: torch.Tensor | None
    beta: torch.Tensor | None
    beta_fired: bool
    replanned: bool
    action_unstretched: torch.Tensor | None = None


class LowLevelChunkExecutor:
    """
    Cache + cursor for π_low native chunks.

    Call ``step`` once per sim step. When ``beta_fired`` is true the caller should
    end the current low-level segment (π_high resamples ω in hierarchical eval).
  """

    def __init__(self, pi_low: Any, config: HierarchicalPolicyConfig) -> None:
        self.pi_low = pi_low
        self.config = config
        self._cached_action: torch.Tensor | None = None
        self._cached_action_pred: torch.Tensor | None = None
        self._cached_action_native: torch.Tensor | None = None
        self._cursor: int = 0
        self._option_step_count: int = 0  # steps since π_high last resampled ω
        self._last_beta: torch.Tensor | None = None
        self._last_replanned: bool = False
        self._last_chunk_was_short: bool = False  # fallback when no duration channel
        self._continuous_horizon: float | None = None  # mean(duration) * horizon, pre-rounding

    @property
    def last_beta(self) -> torch.Tensor | None:
        return self._last_beta

    @property
    def last_replanned(self) -> bool:
        return self._last_replanned

    @property
    def option_done(self) -> bool:
        """True when fewer than 1 step of continuous predicted horizon remains at current cursor.

        duration_termination uses this at every needs_replan boundary: if the duration
        channel says < 1 step is left (e.g. continuous_horizon=2.9, cursor=2 → 0.9
        remaining), the option is considered complete and π_high should resample ω.
        Falls back to the discrete chunk_len < n_action_steps check when no duration
        channel is present.
        """
        if self._continuous_horizon is not None:
            return (self._continuous_horizon - self._cursor) < 1.0
        return self._last_chunk_was_short

    @property
    def cached_action_pred(self) -> torch.Tensor | None:
        return self._cached_action_pred

    @property
    def chunk_cursor(self) -> int:
        return self._cursor

    @property
    def option_step_count(self) -> int:
        """Steps executed since π_high last resampled ω (resets on clear_cache)."""
        return self._option_step_count

    def reset(self) -> None:
        self.clear_cache()
        reset = getattr(self.pi_low, "reset", None)
        if callable(reset):
            reset()

    def clear_cache(self) -> None:
        self._cached_action = None
        self._cached_action_pred = None
        self._cached_action_native = None
        self._cursor = 0
        self._option_step_count = 0  # new option: reset progress counter
        self._last_beta = None
        self._last_replanned = False
        self._last_chunk_was_short = False
        self._continuous_horizon = None

    def _strip_env_actions(self, action: torch.Tensor) -> torch.Tensor:
        dim = int(self.config.env_action_dim)
        if action.shape[-1] <= dim:
            return action
        return action[..., :dim]

    def _check_beta(
        self, obs_dict: dict[str, Any], option_id: torch.Tensor
    ) -> torch.Tensor | None:
        if self._cached_action_pred is None:
            return None
        horizon = max(int(getattr(self.pi_low, "horizon", 1)), 1)
        cursor = min(self._option_step_count / horizon, 1.0)
        beta = self.pi_low.predict_beta(
            obs_dict,
            option_id,
            self._cached_action_pred,
            diffusion_t=self.config.beta_diffusion_t,
            chunk_cursor=cursor,
        )
        self._last_beta = beta
        return beta

    def _beta_fired(self, beta: torch.Tensor | None) -> bool:
        if beta is None:
            return False
        return bool((beta > self.config.beta_transition).any())

    def _compute_continuous_horizon(self) -> float | None:
        """Mean duration channel × horizon (continuous, before rounding to int steps)."""
        pred = self._cached_action_pred
        if pred is None:
            return None
        try:
            if pred.ndim == 3:
                pred = pred[0]  # (T, D+1)
            if pred.ndim != 2 or pred.shape[-1] < 2:
                return None
            d_norm = float(pred[:, -1].mean())
            horizon = int(getattr(self.pi_low, "horizon", pred.shape[0]))
            return d_norm * horizon
        except Exception:
            return None

    def _run_diffusion(self, obs_dict: dict[str, Any], option_id: torch.Tensor) -> None:
        out = self.pi_low.predict_action(obs_dict, option_id)
        self._cached_action = out["action"]
        self._cached_action_pred = out["action_pred"]
        self._cached_action_native = out.get("action_unstretched")
        self._cursor = 0
        self._last_replanned = True
        chunk_len = self._chunk_length()
        self._last_chunk_was_short = (
            chunk_len is not None and chunk_len < self.config.n_action_steps
        )
        self._continuous_horizon = self._compute_continuous_horizon()

    def _chunk_length(self) -> int | None:
        if self._cached_action_native is not None:
            return int(self._cached_action_native.shape[1])
        if self._cached_action is not None:
            return int(self._cached_action.shape[1])
        return None

    def _needs_replan(self) -> bool:
        """Replan when cache is empty or the receding-horizon cap is reached.

        Open-loop: cap = full native chunk length (execute the entire diffused
        sequence before triggering a replan / π_high resample).
        Closed-loop: cap = min(native_len, n_action_steps).
        """
        if self._cached_action is None or self._cached_action_pred is None:
            return True
        chunk_len = self._chunk_length()
        if chunk_len is None:
            return True
        if self.config.open_loop:
            cap = chunk_len  # execute full native decoded chunk before replanning
        else:
            cap = min(chunk_len, int(self.config.n_action_steps))
        return self._cursor >= cap

    @property
    def needs_replan(self) -> bool:
        return self._needs_replan()

    @torch.no_grad()
    def step(self, obs_dict: dict[str, Any], option_id: torch.Tensor) -> LowLevelStepResult:
        """
        Advance one sim step: β check → replan if needed → emit one action row.
        """
        self._last_replanned = False
        replanned = False

        beta = self._check_beta(obs_dict, option_id)
        if self._beta_fired(beta):
            return LowLevelStepResult(
                action=torch.empty(0),
                action_pred=self._cached_action_pred,
                beta=beta,
                beta_fired=True,
                replanned=False,
                action_unstretched=self._cached_action_native,
            )

        if self._needs_replan():
            self._run_diffusion(obs_dict, option_id)
            replanned = True
            # Immediately check β against the fresh plan so a terminal option fires
            # on the same step it is (re-)planned — not one step later.
            beta = self._check_beta(obs_dict, option_id)
            if self._beta_fired(beta):
                return LowLevelStepResult(
                    action=torch.empty(0),
                    action_pred=self._cached_action_pred,
                    beta=beta,
                    beta_fired=True,
                    replanned=True,
                    action_unstretched=self._cached_action_native,
                )

        assert self._cached_action is not None
        cursor = self._cursor
        if self._cached_action_native is not None:
            action = self._cached_action_native[:, cursor : cursor + 1]
        else:
            action = self._strip_env_actions(self._cached_action)[:, cursor : cursor + 1]
        self._cursor = cursor + 1
        self._option_step_count += 1

        return LowLevelStepResult(
            action=action,
            action_pred=self._cached_action_pred,
            beta=beta,
            beta_fired=False,
            replanned=replanned,
            action_unstretched=self._cached_action_native,
        )
