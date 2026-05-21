"""
High-level option policy π_high(ω | s) via flow matching (project_plan §2, §7 row 15).

Trained on **segment-start** observations: given state at the beginning of each
option segment in expert demos, predict which discrete option ω was active.

Not used at eval for frozen DP baseline; wired later via ``HierarchicalController``.

References
----------
- ``project_plan.md`` §2 (π_high), §8B (FM design choices)
- ``soda/dataset/option_aware_dataset.py`` — segment index + ``get_option_segment_bounds``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# TODO §7 row 15: import torch, torch.nn as nn


@dataclass
class HighPolicyConfig:
    """
    Hyperparameters for ``HighPolicy`` (Hydra / YAML should populate this).

    TODO
    ----
    - [ ] ``num_options`` — from zarr (unique ``option_id_*`` values) or config
    - [ ] ``obs_encoder`` — reuse DP vision encoder vs lightweight CNN (§8B)
    - [ ] ``option_embed_dim`` — if FM target is continuous embed + round (§8B)
    - [ ] ``fm_hidden_dim``, ``fm_num_layers``, ``time_embed_dim``
    - [ ] ``num_inference_steps`` — FM Euler steps at inference (§8B sampling)
    """

    num_options: int = 8
    option_embed_dim: int = 64
    fm_hidden_dim: int = 256
    fm_num_layers: int = 4
    time_embed_dim: int = 64
    num_inference_steps: int = 10


class HighPolicy:
    """
    Flow-matching model over option labels (discrete ω).

    Training API (implement)
    ------------------------
    - ``encode_obs(obs_dict) -> global_feat`` — same obs layout as DP (image, agent_pos)
    - ``compute_fm_loss(global_feat, option_id) -> scalar``
    - ``sample_option(global_feat) -> int`` — inference: integrate FM → ω

    TODO §7 row 15 — architecture
    -----------------------------
    - [ ] **Target space (§8B):** start with discrete ω → one-hot or learned
          ``nn.Embedding(num_options, option_embed_dim)`` as flow target
    - [ ] **Conditioning (§8B):** ``s`` only at segment start (locked for P0);
          optional low-level context later
    - [ ] **Flow network:** MLP or small Transformer on ``[global_feat, x_t, t_embed]``
          predicting velocity field ``v_θ(x_t, t | s)``
    - [ ] **Time sampling:** ``t ~ Uniform(0, 1)``; linear path ``x_t = (1-t)*x_0 + t*x_1``
          (or OT path per your FM reference)
    - [ ] **Loss:** MSE ``|| v_θ - (x_1 - x_0) ||^2`` — implement in
          ``losses_high.flow_matching_option_loss`` or here as ``compute_fm_loss``

    TODO §7 row 15 — inference
    --------------------------
    - [ ] ``sample_option``: fixed-step Euler from noise → option embedding → argmax
          or nearest embed to pick discrete ω
    - [ ] Export ``option_id`` int for ``HierarchicalController`` (row 16)
    """

    def __init__(self, cfg: HighPolicyConfig) -> None:
        self.cfg = cfg
        raise NotImplementedError("TODO §7 row 15: build obs encoder + flow network")

    def encode_obs(self, obs: dict[str, Any]) -> Any:
        """
        Map DP-style obs dict to a global feature vector.

        Parameters
        ----------
        obs
            ``image``: ``(B, T, 3, H, W)`` or ``(B, 3, H, W)``;
            ``agent_pos``: ``(B, T, 2)`` or ``(B, 2)``.

        TODO
        ----
        - [ ] Align with low-level obs normalization (shared normalizer stats)
        - [ ] For π_high training, use **last** frame of ``n_obs_steps`` window
              at segment start (or single frame)
        """
        raise NotImplementedError("TODO §7 row 15: encode_obs")

    def compute_fm_loss(
        self,
        global_feat: Any,
        option_id: Any,
    ) -> Any:
        """
        Flow-matching loss for one batch of segment-start states.

        TODO
        ----
        - [ ] Embed ``option_id`` → ``x_1`` target
        - [ ] Sample ``t``, build ``x_t``, predict velocity, MSE vs ``x_1 - x_0``
        - [ ] Delegate to ``soda.training.losses_high.flow_matching_option_loss``
        """
        raise NotImplementedError("TODO §7 row 15: compute_fm_loss")

    def sample_option(self, global_feat: Any) -> Any:
        """
        Sample discrete option ω at inference.

        Returns
        -------
        option_id
            Integer(s) shape ``(B,)``.

        TODO
        ----
        - [ ] Integrate FM ODE / Euler for ``num_inference_steps``
        - [ ] Map continuous endpoint to nearest option embedding or softmax head
        """
        raise NotImplementedError("TODO §7 row 15: sample_option")

    def forward(self, obs: dict[str, Any], option_id: Any | None = None) -> Any:
        """
        Training: pass ``option_id`` → loss scalar.
        Inference: ``option_id is None`` → sampled ω.
        """
        feat = self.encode_obs(obs)
        if option_id is None:
            return self.sample_option(feat)
        return self.compute_fm_loss(feat, option_id)
