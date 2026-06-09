"""State encoder for low-dim Push-T input.

`hssm_rl.EnvModel` takes three nn.Module hooks: `action_encoder`, `encoder`,
`decoder`. For discrete actions we reuse upstream `GridActionEncoder` (an
embedding table) and `GridDecoder` (MLP → action logits). The state encoder
is the only piece we write ourselves — upstream's `CompILEGridEncoder` is a
ConvNet over 12×6×6 grids and doesn't fit our 5-dim flat state.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """MLP embedder for low-dim continuous state vectors.

    Mirrors the output shape contract of upstream `CompILEGridEncoder`:
    `(batch, seq, state_size) -> (batch, seq, output_dim)`.
    """

    def __init__(self, state_size: int = 5, output_dim: int = 128, feat_size: int = 128):
        super().__init__()
        # hssm_rl.py:54 reads `.embedding_size` off the obs encoder to size the
        # combine_action_obs Linear; must match the output channel of forward().
        self.embedding_size = output_dim
        self.net = nn.Sequential(
            nn.Linear(state_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, feat_size),
            nn.ReLU(),
            nn.Linear(feat_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
