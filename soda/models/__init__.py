"""SODA policy modules (π_low, π_high, termination)."""

from soda.models.high_policy import (
    HighPolicy,
    HighPolicyConfig,
    OptionFlowMatchingSchedule,
    OptionVelocityMLP,
    ObsEncoder,
)
from soda.models.low_policy import LowPolicyConfig
from soda.models.termination_head import (
    TerminationHead,
    TerminationHeadConfig,
    termination_bce_from_logits,
)

__all__ = [
    "HighPolicy",
    "HighPolicyConfig",
    "LowPolicy",
    "LowPolicyConfig",
    "OptionFlowMatchingSchedule",
    "OptionVelocityMLP",
    "ObsEncoder",
    "TerminationHead",
    "TerminationHeadConfig",
    "termination_bce_from_logits",
]


def __getattr__(name: str):
    if name == "LowPolicy":
        from soda.models.low_policy import LowPolicy

        return LowPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
