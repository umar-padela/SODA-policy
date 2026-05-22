"""
Hierarchical SODA controller for sim eval (π_high + π_low + termination β).

Mirrors the DP ``BaseImagePolicy.predict_action`` interface so ``soda_runner`` can
swap the policy object without changing the env loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HierarchicalPolicyConfig:
    """Eval-time wiring (filled from Hydra yaml + checkpoint)."""

    config_name: str = "soda_supervised"
    option_id_key: str = "option_id_supervised"
    n_obs_steps: int = 2
    checkpoint_n_action_steps: int = 8


class HierarchicalPolicy:
    """
    π_high selects options; π_low executes until β fires.

    Scaffold only — ``predict_action`` raises until train + load are implemented.
    """

    def __init__(
        self,
        *,
        device: Any,
        config: HierarchicalPolicyConfig,
        checkpoint: Path | None = None,
    ) -> None:
        self.device = device
        self.dtype = getattr(device, "type", "cuda")  # replaced when torch loads weights
        self.config = config
        self.checkpoint = checkpoint
        self._pi_high = None
        self._pi_low = None

    def reset(self) -> None:
        """Reset option / low-level policy state between episodes."""
        # TODO: reset pi_high flow-matching cache and pi_low diffusion state

    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Return ``action`` and ``action_pred`` like DP (B, Ta, Da).

        Receding runner slices to ``n_action_steps`` before ``env.step``.
        """
        raise NotImplementedError(
            "HierarchicalPolicy.predict_action is not implemented yet. "
            "Train pi_high + pi_low, implement load_soda_policy_and_cfg(), then wire "
            "soda.inference.hierarchical_controller.HierarchicalPolicy."
        )
