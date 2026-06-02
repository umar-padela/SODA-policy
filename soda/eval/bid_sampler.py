"""
Native BID (Bidirectional Decoding) sampler for option-conditioned π_low.

Implements the bidirectional_sampler algorithm from Liu et al. (2024) adapted for
SODA's option-conditioned π_low. Unlike BID's subprocess approach (which assumes a
standard DP interface), this version passes option_id through to π_low.

Algorithm (from third_party/bid_diffusion/diffusion_policy/sampler/multi.py):
1. Sample N chunks from strong π_low(obs, option_id) in one batched call
2. Backward coherence: rank by distance to prior, keep top N//K
3. Sample N chunks from weak π_low(obs, option_id)
4. Forward contrast: minimize dist to strong top-k, maximize dist to weak top-k
5. Final score = ratio * LB + (1-ratio) * (LF_pos - LF_neg)
6. Select argmin; store selected action_pred as prior for next step

Prior resets on option switch (clear_cache → reset()).
"""

from __future__ import annotations

from typing import Any

import torch


class BIDSampler:
    """BID test-time selection wrapper around a SODA option-conditioned π_low."""

    def __init__(
        self,
        pi_low_strong: Any,
        pi_low_weak: Any,
        *,
        n_samples: int = 16,
        n_mode: int = 3,
        decay: float = 0.9,
    ) -> None:
        self._strong = pi_low_strong
        self._weak = pi_low_weak
        self.n_samples = int(n_samples)
        self.n_mode = int(n_mode)
        self.decay = float(decay)
        self._prior: torch.Tensor | None = None  # (B, PH, AD) — previous selected action_pred

    def reset(self) -> None:
        """Call on option switch (clear_cache) to restart backward coherence."""
        self._prior = None

    # ------------------------------------------------------------------
    # Batching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_obs(obs_dict: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
        """Tile obs N times along batch dim for a single batched forward pass."""
        batched: dict[str, torch.Tensor] = {}
        for key, val in obs_dict.items():
            # val shape: (B, ...) — replicate N times along batch
            B = val.shape[0]
            # Use expand + contiguous instead of repeat to avoid copies
            expanded = val.unsqueeze(1).expand(
                (B, n) + val.shape[1:]
            ).reshape((B * n,) + val.shape[1:])
            batched[key] = expanded
        return batched

    @staticmethod
    def _batch_option(option_id: torch.Tensor, n: int) -> torch.Tensor:
        """(B,) → (B*N,) by repeating each element N times."""
        return option_id.unsqueeze(1).expand(-1, n).reshape(-1)

    # ------------------------------------------------------------------
    # BID core
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_chunk(
        self,
        obs_dict: dict[str, torch.Tensor],
        option_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Run BID to select the best action chunk from N samples.

        Returns a dict with 'action' and 'action_pred' matching π_low.predict_action().
        """
        N = self.n_samples
        K = self.n_mode
        B = option_id.shape[0]
        prior = self._prior

        obs_batch = self._batch_obs(obs_dict, N)
        opt_batch = self._batch_option(option_id, N)

        # --- Strong policy: N samples ---
        strong_out = self._strong.predict_action(obs_batch, opt_batch)
        action_pred = strong_out["action_pred"]   # (B*N, PH, PD)  PD includes duration channel
        action = strong_out["action"]             # (B*N, AH, ED)  ED = env action dim (2, no duration)
        PH, PD = action_pred.shape[1], action_pred.shape[2]
        AH, ED = action.shape[1], action.shape[2]

        action_pred_s = action_pred.reshape(B, N, PH, PD)  # (B, N, PH, PD)
        action_s = action.reshape(B, N, AH, ED)             # (B, N, AH, ED)

        # --- Weak policy: N samples ---
        weak_out = self._weak.predict_action(obs_batch, opt_batch)
        action_pred_w = weak_out["action_pred"].reshape(B, N, PH, PD)

        # --- Backward coherence ---
        n_top = max(1, N // K)
        if prior is not None:
            # Overlap region: from n_obs_steps-1 to end of prior
            n_obs = int(getattr(self._strong, "n_obs_steps", 2))
            start = n_obs - 1
            end = prior.shape[1]
            n_overlap = end - start

            # Weighted distance between each strong sample and the prior
            diff_s = action_pred_s[:, :, start:end] - prior.unsqueeze(1)[:, :, start:]
            weights = torch.tensor(
                [self.decay ** i for i in range(n_overlap)],
                dtype=diff_s.dtype,
                device=diff_s.device,
            )
            weights = weights / weights.sum()
            dist_backward = (diff_s.norm(dim=-1) * weights).sum(dim=-1)  # (B, N)

            # Keep top n_top strong samples (closest to prior)
            _, sorted_idx = dist_backward.sort(dim=-1)       # ascending
            top_idx = sorted_idx[:, :n_top]                   # (B, n_top)
            B_idx = torch.arange(B, device=top_idx.device)

            action_pred_s_top = action_pred_s[B_idx.unsqueeze(1), top_idx]  # (B, n_top, PH, AD)
            action_s_top = action_s[B_idx.unsqueeze(1), top_idx]             # (B, n_top, AH, AD)
            dist_back_top = dist_backward[B_idx.unsqueeze(1), top_idx]       # (B, n_top)

            # Same for weak policy
            diff_w = action_pred_w[:, :, start:end] - prior.unsqueeze(1)[:, :, start:]
            dist_back_w = (diff_w.norm(dim=-1) * weights).sum(dim=-1)
            _, sorted_w = dist_back_w.sort(dim=-1)
            w_top_idx = sorted_w[:, :n_top]
            action_pred_w_top = action_pred_w[B_idx.unsqueeze(1), w_top_idx]

            # Ratio balancing backward vs forward (from BID paper eq.)
            ratio = float((PH * self.decay) ** 2 / ((PH * self.decay) ** 2 + AH ** 2))
        else:
            # First step — no prior; pure forward contrast
            action_pred_s_top = action_pred_s   # (B, N, PH, AD)
            action_s_top = action_s
            dist_back_top = torch.zeros(B, N, device=action_pred_s.device)
            action_pred_w_top = action_pred_w
            ratio = 0.0
            n_top = N
            B_idx = torch.arange(B, device=action_pred_s.device)

        # --- Forward contrast: pairwise among strong top-k ---
        src = action_pred_s_top.unsqueeze(2)  # (B, n_top, 1, PH, AD)
        tar = action_pred_s_top.unsqueeze(1)  # (B, 1, n_top, PH, AD)
        dist_pos_mat = (src - tar).norm(dim=-1).mean(dim=-1)  # (B, n_top, n_top)

        k_pos = max(1, n_top // 2 + 1)
        vals_pos, _ = dist_pos_mat.topk(min(k_pos, n_top), largest=False, dim=-1)
        dist_avg_pos = vals_pos[:, :, 1:].mean(dim=-1) if vals_pos.shape[-1] > 1 else vals_pos[:, :, 0]

        # Forward contrast: distances to weak top-k
        src_w = action_pred_s_top.unsqueeze(2)    # (B, n_top, 1, PH, AD)
        tar_w = action_pred_w_top.unsqueeze(1)    # (B, 1, n_top, PH, AD)
        dist_neg_mat = (src_w - tar_w).norm(dim=-1).mean(dim=-1)  # (B, n_top, n_top)

        k_neg = max(1, n_top // 2)
        vals_neg, _ = dist_neg_mat.topk(min(k_neg, n_top), largest=False, dim=-1)
        dist_avg_neg = vals_neg.mean(dim=-1)

        # --- Combined score and selection ---
        dist_total = dist_back_top * ratio + (dist_avg_pos - dist_avg_neg) * (1.0 - ratio)
        best_local = dist_total.argmin(dim=-1)  # (B,) index into n_top

        selected_pred = action_pred_s_top[B_idx, best_local]  # (B, PH, PD)
        selected_action = action_s_top[B_idx, best_local]     # (B, AH, ED)

        # Store for next step's backward coherence
        self._prior = selected_pred.detach()

        # Return only the selected chunk — do NOT propagate action_unstretched from
        # the batched strong_out (it has shape B*N x ... and would corrupt the executor cache)
        return {"action": selected_action, "action_pred": selected_pred}


class BIDPolicy:
    """
    BID test-time selection wrapper around a standard (non-option-conditioned) DP.

    Presents the same predict_action(obs_dict) interface as a Columbia DP policy so
    it can be dropped into SodaPushTImageRunner. Maintains backward-coherence prior
    across steps; resets on reset().

    Usage:
        bid = BIDPolicy(strong_dp, weak_dp, n_samples=16, n_mode=3, decay=0.9)
        runner.run(bid)  # replaces runner.run(strong_dp)
    """

    def __init__(
        self,
        strong: Any,
        weak: Any | None,
        *,
        n_samples: int = 16,
        n_mode: int = 3,
        decay: float = 0.9,
    ) -> None:
        self._strong = strong
        self._weak = weak  # None → skip negative contrast (positive-only BID)
        self.n_samples = int(n_samples)
        self.n_mode = int(n_mode)
        self.decay = float(decay)
        self._prior: torch.Tensor | None = None
        # Forward required attributes to the runner
        self.device = strong.device
        self.dtype = getattr(strong, "dtype", torch.float32)
        self.horizon = getattr(strong, "horizon", 16)
        self.n_obs_steps = getattr(strong, "n_obs_steps", 2)
        self.n_action_steps = getattr(strong, "n_action_steps", 1)

    def reset(self) -> None:
        self._prior = None
        if hasattr(self._strong, "reset"):
            self._strong.reset()
        if hasattr(self._weak, "reset"):
            self._weak.reset()

    @torch.no_grad()
    def predict_action(self, obs_dict: dict) -> dict:
        """BID sampling: N chunks from strong + weak, select via BID criteria."""
        N = self.n_samples
        K = self.n_mode
        prior = self._prior

        # Infer batch size from obs
        B = next(iter(obs_dict.values())).shape[0]

        obs_batch = BIDSampler._batch_obs(obs_dict, N)

        # Strong policy: N samples in one forward pass
        strong_out = self._strong.predict_action(obs_batch)
        action_pred = strong_out["action_pred"]       # (B*N, PH, PD)
        action = strong_out["action"]                 # (B*N, AH, ED)
        PH, PD = action_pred.shape[1], action_pred.shape[2]
        AH, ED = action.shape[1], action.shape[2]

        action_pred_s = action_pred.reshape(B, N, PH, PD)
        action_s = action.reshape(B, N, AH, ED)

        # Weak policy: N samples (optional — None → positive-only forward contrast)
        if self._weak is not None:
            weak_out = self._weak.predict_action(obs_batch)
            action_pred_w = weak_out["action_pred"].reshape(B, N, PH, PD)
        else:
            action_pred_w = None

        # --- Backward coherence (same as BIDSampler) ---
        n_top = max(1, N // K)
        B_idx = torch.arange(B, device=action_pred_s.device)

        if prior is not None:
            n_obs = int(getattr(self._strong, "n_obs_steps", 2))
            start = n_obs - 1
            end = prior.shape[1]
            n_overlap = end - start
            weights = torch.tensor(
                [self.decay ** i for i in range(n_overlap)],
                dtype=action_pred_s.dtype, device=action_pred_s.device,
            )
            weights = weights / weights.sum()
            diff_s = action_pred_s[:, :, start:end] - prior.unsqueeze(1)[:, :, start:]
            dist_backward = (diff_s.norm(dim=-1) * weights).sum(dim=-1)
            _, sorted_idx = dist_backward.sort(dim=-1)
            top_idx = sorted_idx[:, :n_top]
            action_pred_s_top = action_pred_s[B_idx.unsqueeze(1), top_idx]
            action_s_top = action_s[B_idx.unsqueeze(1), top_idx]
            dist_back_top = dist_backward[B_idx.unsqueeze(1), top_idx]
            if action_pred_w is not None:
                _, w_sorted = (
                    (action_pred_w[:, :, start:end] - prior.unsqueeze(1)[:, :, start:])
                    .norm(dim=-1).mul(weights).sum(dim=-1).sort(dim=-1)
                )
                action_pred_w_top = action_pred_w[B_idx.unsqueeze(1), w_sorted[:, :n_top]]
            else:
                action_pred_w_top = None
            ratio = float((PH * self.decay) ** 2 / ((PH * self.decay) ** 2 + AH ** 2))
        else:
            action_pred_s_top = action_pred_s
            action_s_top = action_s
            dist_back_top = torch.zeros(B, N, device=action_pred_s.device)
            action_pred_w_top = action_pred_w  # may be None
            ratio = 0.0
            n_top = N

        # --- Forward contrast ---
        src = action_pred_s_top.unsqueeze(2)
        tar = action_pred_s_top.unsqueeze(1)
        dist_pos_mat = (src - tar).norm(dim=-1).mean(dim=-1)
        k_pos = max(1, n_top // 2 + 1)
        vals_pos, _ = dist_pos_mat.topk(min(k_pos, n_top), largest=False, dim=-1)
        dist_avg_pos = vals_pos[:, :, 1:].mean(dim=-1) if vals_pos.shape[-1] > 1 else vals_pos[:, :, 0]

        if action_pred_w_top is not None:
            src_w = action_pred_s_top.unsqueeze(2)
            tar_w = action_pred_w_top.unsqueeze(1)
            dist_neg_mat = (src_w - tar_w).norm(dim=-1).mean(dim=-1)
            k_neg = max(1, n_top // 2)
            vals_neg, _ = dist_neg_mat.topk(min(k_neg, n_top), largest=False, dim=-1)
            dist_avg_neg = vals_neg.mean(dim=-1)
        else:
            dist_avg_neg = torch.zeros_like(dist_avg_pos)

        dist_total = dist_back_top * ratio + (dist_avg_pos - dist_avg_neg) * (1.0 - ratio)
        best_local = dist_total.argmin(dim=-1)
        selected_pred = action_pred_s_top[B_idx, best_local]
        selected_action = action_s_top[B_idx, best_local]

        self._prior = selected_pred.detach()
        return {"action": selected_action, "action_pred": selected_pred}
