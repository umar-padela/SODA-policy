"""Label Push-T demos with a trained LOVE checkpoint.

Writes `data/option_id_unsupervised` into `data/raw/pusht/pusht.zarr` (in
place, mirroring `option_id_supervised`'s schema: int32, chunks=(1000,)).
Run:
    python -m soda.option_discovery.unsupervised.love_adapter.label \\
        --ckpt experiments/love_pusht/best.ckpt

Per-frame label is `out["option_list"]` from `hssm_rl.EnvModel.forward`
(see `third_party/love/hssm_rl.py:463`), which covers the inner `seq_size`
frames of each window. We tile non-overlapping windows with stride
`seq_size` and forward/back-fill the `init_size`-frame edges at episode
boundaries with the nearest covered label.

Rare options (empirical frequency < `min_marginal`) get remapped to the
nearest surviving option id and the surviving set is compacted to `0..K-1`,
so the downstream `num_options` is exactly `K_final` (printed at the end).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

from soda.option_discovery.unsupervised.love_adapter.config import LoveConfig, REPO_ROOT
from soda.option_discovery.unsupervised.love_adapter.dataset import PushtFullEpisodeDataset

LOVE_ROOT = REPO_ROOT / "third_party" / "love"
sys.path.insert(0, str(LOVE_ROOT))

from soda.option_discovery.unsupervised.love_adapter.train import build_model  # noqa: E402


def _cfg_from_ckpt(ckpt: dict) -> LoveConfig:
    """Reconstruct LoveConfig from ckpt; fall back to defaults for any new fields."""
    saved = ckpt.get("cfg", {}) or {}
    cfg = LoveConfig()
    field_types = {f.name: f.type for f in fields(LoveConfig)}
    for k, v in saved.items():
        if k not in field_types:
            continue
        if k in ("zarr_path", "ckpt_dir"):
            v = Path(v)
        setattr(cfg, k, v)
    return cfg


def load_model(ckpt_path: Path, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _cfg_from_ckpt(blob)
    model = build_model(cfg, device)
    model.load_state_dict(blob["model"])
    model.eval()
    centroids = np.asarray(blob["action_centroids"], dtype=np.float32)
    return model, cfg, centroids


@torch.no_grad()
def label_episode(
    model,
    state_np: np.ndarray,
    action_ids_np: np.ndarray,
    cfg: LoveConfig,
    device: torch.device,
) -> np.ndarray:
    T = state_np.shape[0]
    window_len = cfg.window_len
    seq_size = cfg.seq_size
    init_size = cfg.init_size

    if T < window_len:
        return np.zeros(T, dtype=np.int32)

    starts = list(range(0, T - window_len + 1, seq_size))
    if not starts or starts[-1] + window_len < T:
        starts.append(T - window_len)

    obs_batch = np.stack([state_np[s : s + window_len] for s in starts])
    act_batch = np.stack([action_ids_np[s : s + window_len] for s in starts])
    obs_t = torch.from_numpy(obs_batch).to(device)
    act_t = torch.from_numpy(act_batch).long().to(device)

    results = model(obs_t, act_t, seq_size, init_size, cfg.obs_std)
    inner = np.asarray(results["option_list"])  # (B, seq_size) int

    labels = np.full(T, -1, dtype=np.int32)
    for i, s in enumerate(starts):
        a = s + init_size
        b = a + seq_size
        labels[a:b] = inner[i].astype(np.int32)

    # forward / back fill the init_size edges at episode ends
    covered = labels >= 0
    if not covered.any():
        return np.zeros(T, dtype=np.int32)
    first = int(np.argmax(covered))
    last = T - 1 - int(np.argmax(covered[::-1]))
    if first > 0:
        labels[:first] = labels[first]
    if last < T - 1:
        labels[last + 1 :] = labels[last]
    return labels


def filter_degenerate(labels: np.ndarray, min_marginal: float) -> tuple[np.ndarray, int]:
    """Drop rarely-used options; remap to nearest surviving id; compact to 0..K-1."""
    unique, counts = np.unique(labels, return_counts=True)
    freq = counts / counts.sum()
    keep = sorted({int(u) for u, f in zip(unique, freq) if f >= min_marginal})
    if not keep:
        return labels.astype(np.int32), int(unique.max() + 1)

    out = labels.copy()
    for u in unique:
        u = int(u)
        if u in keep:
            continue
        nearest = min(keep, key=lambda k: abs(k - u))
        out[labels == u] = nearest

    remap = {old: new for new, old in enumerate(sorted(set(out.tolist())))}
    out = np.vectorize(remap.get)(out).astype(np.int32)
    return out, len(remap)


def write_labels(zarr_path: Path, labels: np.ndarray) -> None:
    root = zarr.open(str(zarr_path), mode="a")
    data = root["data"]
    if "option_id_unsupervised" in data:
        data["option_id_unsupervised"][:] = labels
        print(f"updated data/option_id_unsupervised in {zarr_path}")
    else:
        data.array(
            "option_id_unsupervised",
            labels,
            chunks=(1000,),
            dtype="int32",
        )
        print(f"created data/option_id_unsupervised in {zarr_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--zarr", type=Path, default=LoveConfig().zarr_path)
    p.add_argument("--min-marginal", type=float, default=LoveConfig().min_marginal)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.ckpt} on {device}")
    model, cfg, centroids = load_model(args.ckpt, device)

    ds = PushtFullEpisodeDataset(args.zarr, action_centroids=centroids)
    total = ds.spans[-1][1]
    labels = np.zeros(total, dtype=np.int32)
    print(f"labeling {len(ds)} episodes ({total} frames total)")
    for i in range(len(ds)):
        _obs, _act, start, end = ds[i]
        state_np = ds.state[start:end]
        action_ids_np = ds.action_ids[start:end]
        z = label_episode(model, state_np, action_ids_np, cfg, device)
        labels[start:end] = z
        if i % 20 == 0:
            print(f"  labeled episode {i}/{len(ds)}")

    unique_pre, counts_pre = np.unique(labels, return_counts=True)
    print("pre-filter label distribution:")
    for u, c in zip(unique_pre, counts_pre):
        print(f"  {u}: {c}  ({c / counts_pre.sum():.3%})")

    labels, k_final = filter_degenerate(labels, args.min_marginal)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"final label distribution (K_final={k_final}):")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}  ({c / counts.sum():.3%})")

    write_labels(args.zarr, labels)
    print(f"K_final={k_final} — update num_options in configs/pusht/soda_unsupervised.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
