"""
Gym 0.23 compatibility for Columbia diffusion_policy vector envs.

DP was written for gym 0.21 where ``concatenate(items, out, space)``.
Gym 0.23 uses ``concatenate(space, items, out)``.
"""

from __future__ import annotations

_PATCHED = False


def apply_gym_dp_vector_compat() -> None:
    """Patch vector utils once before DP AsyncVectorEnv reset/step."""
    global _PATCHED
    if _PATCHED:
        return

    from gym.spaces import Space
    import gym.vector.utils as gym_vector_utils

    _orig = gym_vector_utils.concatenate

    def concatenate_dp_compat(first, second, third):
        if isinstance(third, Space):
            items = list(first) if isinstance(first, tuple) else first
            return _orig(third, items, second)
        return _orig(first, second, third)

    gym_vector_utils.concatenate = concatenate_dp_compat

    # DP submodules bind concatenate at import time — patch those namespaces too.
    from diffusion_policy.gym_util import async_vector_env, sync_vector_env

    async_vector_env.concatenate = concatenate_dp_compat
    sync_vector_env.concatenate = concatenate_dp_compat

    _PATCHED = True
