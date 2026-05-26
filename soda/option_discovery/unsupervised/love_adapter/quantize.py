"""Discretize continuous 2D Push-T actions into a small codebook.

`hssm_rl.EnvModel` reconstructs actions with `F.cross_entropy` (line ~803),
so it only accepts integer action ids. We fit K cluster centroids on the
demo action stream once, quantize every frame's action to its nearest
centroid id, and persist the centroids alongside the checkpoint so labeling
uses the same codebook the model was trained on.

Pure-numpy Lloyd's algorithm — no sklearn dependency (not in environment.yml).
"""
from __future__ import annotations

import numpy as np


def fit_kmeans(
    actions: np.ndarray,
    n_clusters: int,
    n_iters: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """Return centroids of shape (n_clusters, action_dim).

    Random sample initialization + Lloyd iterations until assignments stabilize
    or n_iters elapse.
    """
    rng = np.random.default_rng(seed)
    n, d = actions.shape
    if n_clusters > n:
        raise ValueError(f"n_clusters={n_clusters} exceeds n_samples={n}")

    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centroids = actions[init_idx].astype(np.float32).copy()
    prev_assign = np.full(n, -1, dtype=np.int64)

    for _ in range(n_iters):
        # (n, k) squared distances
        d2 = ((actions[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        assign = d2.argmin(axis=1)
        if np.array_equal(assign, prev_assign):
            break
        prev_assign = assign
        for k in range(n_clusters):
            members = actions[assign == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
            # empty cluster: leave centroid in place (rare with random init)

    return centroids


def quantize(actions: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each action to the nearest centroid id. Returns int64 ids."""
    d2 = ((actions[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    return d2.argmin(axis=1).astype(np.int64)
