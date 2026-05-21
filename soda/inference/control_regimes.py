"""
Closed-loop control parameters for eval (DP baseline and SODA).

action_horizon
    How many actions from the policy output are executed before replanning
    with a fresh observation. Default ``1`` (replan every sim step).

The policy decides how long each prediction is (DP: fixed policy_horizon 16;
SODA: native skill length via duration channel + ``horizon_stretch_max`` at train time).
Eval only slices the first ``action_horizon``
steps from that output.

open regime
    One ``predict_action`` at episode start, then execute the full predicted
    chunk without further replanning (``action_horizon`` is ignored).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ControlRegime = Literal["receding", "open"]


@dataclass(frozen=True)
class ControlParams:
    """Per-rollout execution of a predicted action chunk."""

    action_horizon: int = 1
    regime: ControlRegime = "receding"
    # Index into the predicted trajectory where execution starts (DP: n_obs_steps - 1).
    action_start_index: int = 0

    def __post_init__(self) -> None:
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        if self.action_start_index < 0:
            raise ValueError("action_start_index must be >= 0")


@dataclass(frozen=True)
class SodaControlParams:
    """
    SODA low-level loop: chunk length comes from the policy at runtime;
    ``action_horizon`` is the replan cadence inside a skill segment.
    """

    action_horizon: int = 1
    regime: ControlRegime = "receding"

    def for_skill(
        self,
        available_actions: int,
        *,
        n_obs_steps: int = 1,
    ) -> ControlParams:
        """Build per-call ControlParams; clamp horizon to what the policy returned."""
        start = max(0, n_obs_steps - 1)
        remaining = max(1, available_actions - start)
        return ControlParams(
            action_horizon=min(self.action_horizon, remaining),
            regime=self.regime,
            action_start_index=start,
        )


def resolve_control_params(
    *,
    regime: ControlRegime,
    n_obs_steps: int,
    action_horizon: int = 1,
) -> ControlParams:
    """Build rollout control params (only ``action_horizon`` is user-controlled)."""
    return ControlParams(
        action_horizon=action_horizon,
        regime=regime,
        action_start_index=max(0, n_obs_steps - 1),
    )


def extract_execute_actions(
    action_dict: dict[str, Any],
    control: ControlParams,
) -> Any:
    """
    Slice a policy output to the actions to pass to ``env.step``.

    Uses the full ``action_pred`` when present (DP) so ``action_horizon`` can
    differ from the checkpoint's baked-in ``n_action_steps``.
    """
    import numpy as np

    start = control.action_start_index

    if "action_pred" in action_dict:
        pred = action_dict["action_pred"]
        if hasattr(pred, "detach"):
            pred = pred.detach().cpu().numpy()
        else:
            pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[0]
        available = pred.shape[0] - start
        if control.regime == "open":
            n_exec = available
        else:
            n_exec = control.action_horizon
            if n_exec > available:
                raise ValueError(
                    f"action_horizon ({n_exec}) exceeds available actions "
                    f"({available}) after start index {start}"
                )
        execute = pred[start : start + n_exec]
    else:
        action = action_dict["action"]
        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()
        else:
            action = np.asarray(action)
        if action.ndim == 3:
            action = action[0]
        available = action.shape[0]
        if control.regime == "open":
            execute = action
        else:
            if control.action_horizon > available:
                raise ValueError(
                    f"action_horizon ({control.action_horizon}) exceeds "
                    f"available actions ({available})"
                )
            execute = action[: control.action_horizon]

    if control.regime != "open" and execute.shape[0] != control.action_horizon:
        raise ValueError(
            f"Expected {control.action_horizon} actions, got shape {execute.shape}"
        )
    return execute
