"""
Hierarchical SODA controller for sim eval (π_high + π_low + termination β).

Mirrors the DP ``BaseImagePolicy.predict_action`` interface. Training produces a
full action chunk (variable horizon for SODA); **eval** chooses how many rows to
execute before replanning via ``HierarchicalPolicyConfig.n_action_steps`` (default
8 = receding-horizon control, same protocol as vanilla DP).

Each sim step:

1. **β (every step, once a plan exists):** ``pi_low.predict_beta`` on the cached
   ``action_pred`` at ``t=0`` with current obs + ω → ``TerminationHead``.
2. **Replan** when β fires, the cache is empty, or
   ``min(native_len, n_action_steps)`` native rows have been executed.
3. **Return** the next row from π_low's cached ``action_unstretched`` chunk.

β is evaluated on **every** sim step while executing a cached chunk; diffusion
replanning happens at most every ``n_action_steps`` steps unless β fires earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch


@dataclass
class HierarchicalPolicyConfig:
    """Eval-time wiring (Hydra yaml + checkpoint loader)."""

    config_name: str = "soda_supervised"
    option_id_key: str = "option_id_supervised"
    env_action_dim: int = 2
    n_action_steps: int = 8  # eval-only receding-horizon execute window before replan
    beta_transition: float = 0.5
    beta_diffusion_t: int = 0


class HierarchicalPolicy:
    """π_high selects ω; π_low diffuses chunks; β may end a segment early."""

    def __init__(
        self,
        *,
        device: Any,
        config: HierarchicalPolicyConfig,
        checkpoint: Path | None = None,
        pi_high: Any | None = None,
        pi_low: Any | None = None,
    ) -> None:
        self.device = device
        self.dtype = torch.float32
        self.config = config
        self.checkpoint = checkpoint
        self._pi_high = pi_high
        self._pi_low = pi_low

        self._current_option_id: torch.Tensor | None = None
        self._cached_action: torch.Tensor | None = None
        self._cached_action_pred: torch.Tensor | None = None
        self._cached_action_native: torch.Tensor | None = None  # (B, native_steps, D)
        self._cursor: int = 0
        self._last_beta: torch.Tensor | None = None

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

    def reset(self) -> None:
        self._current_option_id = None
        self._cached_action = None
        self._cached_action_pred = None
        self._cached_action_native = None
        self._cursor = 0
        self._last_beta = None
        if self._pi_high is not None and hasattr(self._pi_high, "reset"):
            self._pi_high.reset()
        if self._pi_low is not None and hasattr(self._pi_low, "reset"):
            self._pi_low.reset()

    def _require_policies(self) -> tuple[Any, Any]:
        if self._pi_high is None or self._pi_low is None:
            raise NotImplementedError(
                "Attach pi_high and pi_low on HierarchicalPolicy before predict_action."
            )
        return self._pi_high, self._pi_low

    def _n_action_steps(self, pi_low: Any) -> int:
        _ = pi_low
        return max(1, int(self.config.n_action_steps))

    def _strip_env_actions(self, action: torch.Tensor) -> torch.Tensor:
        dim = int(self.config.env_action_dim)
        if action.shape[-1] <= dim:
            return action
        return action[..., :dim]

    def _sample_option(self, pi_high: Any, obs_dict: dict[str, Any]) -> torch.Tensor:
        return pi_high.sample_option(pi_high.encode_obs(obs_dict))

    def _check_beta(
        self, pi_low: Any, obs_dict: dict[str, Any], option_id: torch.Tensor
    ) -> torch.Tensor | None:
        if self._cached_action_pred is None:
            return None
        beta = pi_low.predict_beta(
            obs_dict,
            option_id,
            self._cached_action_pred,
            diffusion_t=self.config.beta_diffusion_t,
        )
        self._last_beta = beta
        return beta

    def _beta_fired(self, beta: torch.Tensor | None) -> bool:
        if beta is None:
            return False
        return bool((beta > self.config.beta_transition).any())

    def _run_diffusion(
        self,
        pi_low: Any,
        obs_dict: dict[str, Any],
        option_id: torch.Tensor,
    ) -> None:
        out = pi_low.predict_action(obs_dict, option_id)
        self._cached_action = out["action"]
        self._cached_action_pred = out["action_pred"]
        self._cached_action_native = out.get("action_unstretched")  # (B, native_steps, D)
        self._cursor = 0

    def _needs_diffusion(self, pi_low: Any, beta: torch.Tensor | None) -> bool:
        if self._cached_action is None or self._cached_action_pred is None:
            return True
        if self._beta_fired(beta):
            return True
        if self._cached_action_native is not None:
            native_len = int(self._cached_action_native.shape[1])
            cap = min(native_len, self._n_action_steps(pi_low))
            return self._cursor >= cap
        # Fallback: fixed receding-horizon window (P0 protocol).
        if self._cursor >= self._n_action_steps(pi_low):
            return True
        if self._cursor >= int(self._cached_action.shape[1]):
            return True
        return False

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Return the next single env action from the hierarchical stack.

        ``soda_runner`` calls this once per sim step. Replan window size comes
        from ``config.n_action_steps`` (eval yaml; default 8).
        """
        pi_high, pi_low = self._require_policies()

        if self._current_option_id is None:
            self._current_option_id = self._sample_option(pi_high, obs_dict)

        beta = self._check_beta(pi_low, obs_dict, self._current_option_id)
        if self._beta_fired(beta):
            self._current_option_id = self._sample_option(pi_high, obs_dict)
            self._cached_action = None
            self._cached_action_pred = None
            self._cached_action_native = None

        if self._needs_diffusion(pi_low, beta):
            self._run_diffusion(pi_low, obs_dict, self._current_option_id)

        assert self._cached_action is not None
        cursor = self._cursor
        if self._cached_action_native is not None:
            # Execute from the decompressed native-time chunk (variable-horizon path).
            action = self._cached_action_native[:, cursor : cursor + 1]
        else:
            # Fallback: fixed receding-horizon from the stretched prediction window.
            action = self._strip_env_actions(self._cached_action)[:, cursor : cursor + 1]
        self._cursor = cursor + 1

        result: dict[str, Any] = {
            "action": action,
            "action_pred": self._cached_action_pred,
        }
        if self._last_beta is not None:
            result["beta"] = self._last_beta
        return result
