"""LoveConfig — all tunable knobs for the LOVE adapter in one dataclass.

Defaults are seeded from upstream `third_party/love/train_rl.py` grid_world
recipe, adjusted for Push-T's 5-dim state and (discretized) 2-dim action.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# __file__: soda/option_discovery/unsupervised/love_adapter/config.py
# parents:    4         3                  2             1             0
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"
DEFAULT_CKPT_DIR = REPO_ROOT / "experiments" / "love_pusht"


@dataclass
class LoveConfig:
    # paths
    zarr_path: Path = DEFAULT_ZARR
    ckpt_dir: Path = DEFAULT_CKPT_DIR

    # observation / action sizes
    state_size: int = 5            # Push-T state dim (gripper xy, block xy, block rot)
    num_action_bins: int = 16      # k-means codebook size for discretizing actions

    # LOVE model sizes (upstream grid_world defaults)
    belief_size: int = 128
    num_layers: int = 2
    latent_n: int = 10             # upper bound on # of options; filtered at label time
    seg_len: int = 100             # max segment length (frames)
    seg_num: int = 100             # max segments per episode

    # training window
    seq_size: int = 6              # inner sequence length consumed by hssm_rl
    init_size: int = 1             # init/burn-in frames at each end
    # window_len = seq_size + 2 * init_size  (computed via property below)

    # LOVE objective coefficients
    coding_len_coeff: float = 0.005
    kl_coeff: float = 0.0
    rec_coeff: float = 1.0
    use_abs_pos_kl: bool = True
    use_min_length_boundary_mask: bool = True
    ddo: bool = False
    output_normal: bool = True
    obs_std: float = 1.0

    # gumbel / boundary annealing (upstream defaults)
    max_beta: float = 1.0
    min_beta: float = 0.1
    beta_anneal: float = 100.0

    # optimization
    batch_size: int = 64
    learn_rate: float = 5e-4
    grad_clip: float = 10.0
    max_iters: int = 20000
    seed: int = 0

    # labeling-time filter
    min_marginal: float = 0.005    # drop options with empirical freq below this

    @property
    def window_len(self) -> int:
        return self.seq_size + 2 * self.init_size
