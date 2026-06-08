"""
Action-space noise for robustness evaluation.

Two implementations:

VelocityProportionalNoise (default / recommended)
    BID-style velocity-proportional noise for position-control actions.
    Differentiates planned positions into velocities, adds i.i.d. multiplicative
    noise to each velocity, then integrates back to positions:

        vel[t]       = action[t] - prev_executed_pos
        noisy_vel[t] = vel[t] * (1 + η · N(0,1))
        noisy_pos[t] = prev_executed_pos + noisy_vel[t]

    Equivalent to BID's formula for velocity-control environments:
        noisy_pos[t] = pos[t] + η · cumsum(noise_seed · velocity)[t]

    η is naturally scale-free: η=1 means ±100% velocity noise. No action_scale needed.
    prev_executed_pos is tracked across apply_to_chunk calls; call reset() each episode.
    Call set_initial_pos(agent_pos) after reset() for accurate first-step noise.

TemporallyCorrelatedNoise (legacy AR(1))
    AR(1) process:  z_t = ρ·z_{t-1} + √(1-ρ²)·ε_t,  ε_t ~ N(0,I)
    Executed action: a_t + η·z_t

    Pass ``action_scale`` (per-dim, same units as the action) so that η is
    interpreted in normalised space. Runners derive this from the policy normalizer:
        action_scale = 1 / policy.normalizer['action'].params_dict['scale']
    which equals (max − min) / 2 per dimension for limits-normalised actions.

η=0.0 → identity / no overhead in both classes.
"""

from __future__ import annotations

import numpy as np


class VelocityProportionalNoise:
    """
    BID-style velocity-proportional noise for position-control actions.

    Tracks the last executed position across apply_to_chunk calls within an
    episode. Call reset() at episode start, then set_initial_pos(agent_pos)
    so the very first step also receives correct noise.

    Supports shapes:
      (n_envs, n_steps, action_dim)  — vectorised runner
      (n_steps, action_dim)          — single-env serial runner
    """

    def __init__(self, eta: float, n_envs: int = 1, action_dim: int = 2):
        self.eta = float(eta)
        self.n_envs = int(n_envs)
        self.action_dim = int(action_dim)
        self._prev_pos: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_pos = None

    def set_initial_pos(self, pos: np.ndarray) -> None:
        """Set the agent's starting position so the first velocity is correct.

        Accepts (n_envs, action_dim) or (n_envs, n_obs_steps, action_dim) —
        takes the last time step in the latter case.
        """
        pos = np.asarray(pos, dtype=np.float32)
        if pos.ndim == 3:
            pos = pos[:, -1, :]  # (n_envs, n_obs_steps, action_dim) → take latest
        self._prev_pos = pos.reshape(self.n_envs, self.action_dim)

    def apply_to_chunk(self, action: np.ndarray) -> np.ndarray:
        if self.eta == 0.0:
            return action
        action = action.copy()
        squeezed = action.ndim == 2
        if squeezed:
            action = action[np.newaxis]  # (1, n_steps, action_dim)
        n_envs, n_steps, action_dim = action.shape

        if self._prev_pos is None:
            # Fallback when set_initial_pos was not called: treat first vel as zero.
            self._prev_pos = action[:, 0, :].copy()

        noisy = np.empty_like(action)
        prev = self._prev_pos  # (n_envs, action_dim)

        for t in range(n_steps):
            vel = action[:, t, :] - prev
            noise_seed = np.random.randn(n_envs, action_dim).astype(np.float32)
            noisy_vel = vel * (1.0 + self.eta * noise_seed)
            noisy[:, t, :] = prev + noisy_vel
            prev = noisy[:, t, :]

        self._prev_pos = prev.copy()

        if squeezed:
            noisy = noisy[0]
        return noisy


class TemporallyCorrelatedNoise:
    """Per-env AR(1) noise; reset at episode start, apply before env.step()."""

    def __init__(
        self,
        eta: float,
        rho: float = 0.9,
        n_envs: int = 1,
        action_dim: int = 2,
        action_scale: np.ndarray | None = None,
    ):
        self.eta = float(eta)
        self.rho = float(rho)
        self.n_envs = int(n_envs)
        self.action_dim = int(action_dim)
        self._state = np.zeros((self.n_envs, self.action_dim), dtype=np.float32)
        # (1, action_dim) broadcast shape; ones = raw action-space units (legacy)
        if action_scale is not None:
            self._action_scale = np.asarray(action_scale, dtype=np.float32).reshape(1, -1)
        else:
            self._action_scale = np.ones((1, self.action_dim), dtype=np.float32)

    def reset(self) -> None:
        self._state[:] = 0.0

    def apply_to_chunk(self, action: np.ndarray, scale_by_magnitude: bool = False) -> np.ndarray:
        """
        Add temporally correlated noise to an action chunk.

        Noise magnitude per step: η × action_scale × AR1_state
        With action_scale from the normalizer, η=1.0 ≈ 1 normalised unit of noise.

        scale_by_magnitude=True additionally scales each step's noise by the
        L2 norm of that step's action (after action_scale is applied).

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
            scaled_noise = self.eta * self._action_scale * self._state  # (n_envs, action_dim)
            if scale_by_magnitude:
                magnitude = np.linalg.norm(action[:, t, :], axis=-1, keepdims=True)
                action[:, t, :] += magnitude * scaled_noise
            else:
                action[:, t, :] += scaled_noise
        if squeezed:
            action = action[0]
        return action
