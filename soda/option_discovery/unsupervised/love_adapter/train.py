"""Train LOVE on Push-T state demonstrations.

Imports `EnvModel` from `third_party/love/hssm_rl.py` (matching upstream
`train_rl.py`, NOT `hssm_v2.py`). Reuses upstream `GridActionEncoder` /
`GridDecoder`; uses our own `StateEncoder` for the 5-dim state.

Run:
    python -m soda.option_discovery.unsupervised.love_adapter.train

Outputs (under `experiments/love_pusht/`):
    best.ckpt   — best train_loss on val split (state_dict + cfg + centroids)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from soda.option_discovery.unsupervised.love_adapter.config import LoveConfig, REPO_ROOT
from soda.option_discovery.unsupervised.love_adapter.dataset import PushtLoveDataset
from soda.option_discovery.unsupervised.love_adapter.encoders import StateEncoder


def _import_love():
    """Lazy-import LOVE upstream. Pulls in wandb/gym/PIL — only do this when training."""
    love_root = REPO_ROOT / "third_party" / "love"
    if str(love_root) not in sys.path:
        sys.path.insert(0, str(love_root))
    from hssm_rl import EnvModel  # noqa: E402
    from modules import GridActionEncoder, GridDecoder  # noqa: E402

    return EnvModel, GridActionEncoder, GridDecoder


def build_model(cfg: LoveConfig, device: torch.device):
    EnvModel, GridActionEncoder, GridDecoder = _import_love()
    action_encoder = GridActionEncoder(
        action_size=cfg.num_action_bins, embedding_size=cfg.belief_size
    )
    encoder = StateEncoder(
        state_size=cfg.state_size, output_dim=cfg.belief_size, feat_size=cfg.belief_size
    )
    decoder = GridDecoder(
        input_size=cfg.belief_size,
        action_size=cfg.num_action_bins,
        feat_size=cfg.belief_size,
    )
    model = EnvModel(
        action_encoder=action_encoder,
        encoder=encoder,
        decoder=decoder,
        belief_size=cfg.belief_size,
        state_size=cfg.state_size,
        num_layers=cfg.num_layers,
        max_seg_len=cfg.seg_len,
        max_seg_num=cfg.seg_num,
        latent_n=cfg.latent_n,
        kl_coeff=cfg.kl_coeff,
        rec_coeff=cfg.rec_coeff,
        use_abs_pos_kl=cfg.use_abs_pos_kl,
        coding_len_coeff=cfg.coding_len_coeff,
        use_min_length_boundary_mask=cfg.use_min_length_boundary_mask,
        ddo=cfg.ddo,
        output_normal=cfg.output_normal,
    ).to(device)
    return model


def _step_anneal(model, cfg: LoveConfig, b_idx: int) -> None:
    if cfg.beta_anneal:
        model.state_model.mask_beta = (cfg.max_beta - cfg.min_beta) * (
            0.999 ** (b_idx / cfg.beta_anneal)
        ) + cfg.min_beta
    else:
        model.state_model.mask_beta = cfg.max_beta


def _adapt_coding_len(model, cfg: LoveConfig, results: dict, b_idx: int) -> None:
    """Upstream's adaptive coding_len_coeff update (train_rl.py lines ~222-230)."""
    if cfg.coding_len_coeff <= 0:
        return
    if results["obs_cost"].mean().item() < 0.02:
        model.coding_len_coeff += 0.00002
    elif b_idx > 0:
        model.coding_len_coeff -= 0.00002
    model.coding_len_coeff = min(0.05, model.coding_len_coeff)
    model.coding_len_coeff = max(0.0, model.coding_len_coeff)


@torch.no_grad()
def _eval(model, dl: DataLoader, cfg: LoveConfig, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for obs, act in dl:
        obs = obs.to(device)
        act = act.to(device)
        results = model(obs, act, cfg.seq_size, cfg.init_size, cfg.obs_std)
        bs = obs.size(0)
        total += float(results["train_loss"].item()) * bs
        n += bs
    model.train()
    return total / max(n, 1)


def train(cfg: LoveConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PushtLoveDataset(
        cfg.zarr_path,
        window_len=cfg.window_len,
        num_action_bins=cfg.num_action_bins,
        kmeans_seed=cfg.seed,
    )
    print(
        f"dataset: {len(ds)} windows of length {cfg.window_len}; "
        f"action codebook size {cfg.num_action_bins}"
    )

    n_val = max(1, len(ds) // 10)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_dl = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0
    )
    val_dl = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    model = build_model(cfg, device)
    optimizer = Adam(model.parameters(), lr=cfg.learn_rate, amsgrad=True)

    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = cfg.ckpt_dir / "best.ckpt"

    centroids = ds.action_centroids
    state_mean = ds.state_mean
    state_std = ds.state_std
    np.save(cfg.ckpt_dir / "action_centroids.npy", centroids)
    np.save(cfg.ckpt_dir / "state_mean.npy", state_mean)
    np.save(cfg.ckpt_dir / "state_std.npy", state_std)
    (cfg.ckpt_dir / "config.json").write_text(
        json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, indent=2)
    )

    b_idx = 0
    while b_idx <= cfg.max_iters:
        for obs, act in train_dl:
            b_idx += 1
            _step_anneal(model, cfg, b_idx)

            obs = obs.to(device)
            act = act.to(device)

            model.train()
            optimizer.zero_grad()
            results = model(obs, act, cfg.seq_size, cfg.init_size, cfg.obs_std)
            _adapt_coding_len(model, cfg, results, b_idx)

            loss = results["train_loss"]
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if b_idx % 100 == 0:
                obs_cost = float(results["obs_cost"].mean().item())
                enc_len = float(results["encoding_length"].item())
                print(
                    f"iter {b_idx:6d}  loss {float(loss.item()):.4f}  "
                    f"obs_cost {obs_cost:.4f}  enc_len {enc_len:.4f}  "
                    f"coding_len_coeff {model.coding_len_coeff:.5f}"
                )
            if b_idx % 1000 == 0:
                val = _eval(model, val_dl, cfg, device)
                print(f"iter {b_idx:6d}  val_loss {val:.4f}")
                if val < best_val:
                    best_val = val
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "cfg": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()},
                            "action_centroids": centroids,
                            "state_mean": state_mean,
                            "state_std": state_std,
                            "iter": b_idx,
                            "val_loss": val,
                        },
                        best_path,
                    )
                    print(f"  → new best, saved {best_path}")
            if b_idx >= cfg.max_iters:
                break

    return best_path


if __name__ == "__main__":
    cfg = LoveConfig()
    path = train(cfg)
    print(f"done. best checkpoint: {path}")
