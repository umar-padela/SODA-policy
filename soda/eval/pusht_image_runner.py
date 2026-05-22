"""
Columbia ``PushTImageRunner`` with ``AsyncVectorEnv(shared_memory=False)``.

DP's default ``shared_memory=True`` fails on MultiStepWrapper Dict obs spaces
(OrderedDict batching) in some gym builds — common on Modal. Eval still uses
parallel envs; only shared-memory batching is disabled.
"""

from __future__ import annotations

from soda.eval.gym_vector_compat import apply_gym_dp_vector_compat

apply_gym_dp_vector_compat()

from diffusion_policy.env_runner import pusht_image_runner as _dp_pusht_runner
from diffusion_policy.env_runner.pusht_image_runner import PushTImageRunner as _PushTImageRunner
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv


def _async_vector_env_no_shm(env_fns):
    return AsyncVectorEnv(env_fns, shared_memory=False)


class PushTImageRunner(_PushTImageRunner):
    """Drop-in replacement for diffusion_policy ``PushTImageRunner``."""

    def __init__(self, *args, **kwargs):
        _orig = _dp_pusht_runner.AsyncVectorEnv
        _dp_pusht_runner.AsyncVectorEnv = _async_vector_env_no_shm
        try:
            super().__init__(*args, **kwargs)
        finally:
            _dp_pusht_runner.AsyncVectorEnv = _orig
