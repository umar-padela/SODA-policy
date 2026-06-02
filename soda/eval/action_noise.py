"""
Temporally correlated action-space noise (Liu et al. 2024 / BID).

AR(1) process:  z_t = ρ·z_{t-1} + √(1-ρ²)·ε_t,  ε_t ~ N(0,I)
Executed action: a_t + η·z_t

η=0.0 (default) → identity / no overhead.
"""

from __future__ import annotations

import numpy as np


class TemporallyCorrelatedNoise:
    """Per-env AR(1) noise; reset at episode start, apply before env.step()."""

    def __init__(self, eta: float, rho: float = 0.9, n_envs: int = 1, action_dim: int = 2):
        self.eta = float(eta)
        self.rho = float(rho)
        self.n_envs = int(n_envs)
        self.action_dim = int(action_dim)
        self._state = np.zeros((self.n_envs, self.action_dim), dtype=np.float32)

    def reset(self) -> None:
        self._state[:] = 0.0

    def apply_to_chunk(self, action: np.ndarray, scale_by_magnitude: bool = False) -> np.ndarray:
        """
        Add temporally correlated noise to an action chunk.

        scale_by_magnitude=True matches BID paper: noise scaled by per-step action
        magnitude so η=1.0 means noise std = 1× the action vector's L2 norm.

        Supports shapes:
          (n_envs, n_steps, action_dim)  — vectorised runner
          (n_steps, action_dim)          — single-env serial runner
        """
        if self.eta == 0.0:
            return action
        action = action.copy()
        squeezed = action.ndim == 2
        if squeezed:
            action = action[np.newaxis]  # (1, n_steps, action_dim)
        n_steps = action.shape[1]
        for t in range(n_steps):
            eps = np.random.randn(self.n_envs, self.action_dim).astype(np.float32)
            self._state = (
                self.rho * self._state + np.sqrt(1.0 - self.rho ** 2) * eps
            )
            if scale_by_magnitude:
                magnitude = np.linalg.norm(action[:, t, :], axis=-1, keepdims=True)
                action[:, t, :] += self.eta * magnitude * self._state
            else:
                action[:, t, :] += self.eta * self._state
        if squeezed:
            action = action[0]
        return action
