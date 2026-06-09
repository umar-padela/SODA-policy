#!/usr/bin/env python3
"""Apply a LOVE label .npy (from Modal labeling) into the local pusht.zarr.

Companion to modal/modal_label_love.py. The Modal labeling function writes
`option_id_unsupervised.npy` to the soda-experiments volume; after pulling
that file locally:

    modal volume get soda-experiments love_pusht/option_id_unsupervised.npy ./experiments/love_pusht/
    python scripts/apply_love_labels.py

this script opens `data/raw/pusht/pusht.zarr` and writes the labels into
`data/option_id_unsupervised` (int32, chunks=(1000,)) — same schema as
`option_id_supervised`. Idempotent: re-running just overwrites the array.

Only needs `zarr<3` and `numpy` — no torch / LOVE upstream / Modal in your
local env.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPY = REPO_ROOT / "experiments" / "love_pusht" / "option_id_unsupervised.npy"
DEFAULT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"


def _require_zarr_v2() -> None:
    if not zarr.__version__.startswith("2."):
        raise RuntimeError(
            f"Zarr v2 required (pip install 'zarr<3'). Found {zarr.__version__}"
        )


def apply_labels(npy_path: Path, zarr_path: Path) -> None:
    _require_zarr_v2()
    labels = np.load(npy_path).astype(np.int32)
    root = zarr.open(str(zarr_path), mode="a")
    state_len = root["data"]["state"].shape[0]
    if labels.shape[0] != state_len:
        raise ValueError(
            f"label length {labels.shape[0]} != state length {state_len} — "
            f"the .npy was generated against a different zarr"
        )
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

    unique, counts = np.unique(labels, return_counts=True)
    print(f"label distribution (K_final={len(unique)}):")
    for u, c in zip(unique, counts):
        print(f"  {int(u)}: {c}  ({c / counts.sum():.3%})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npy", type=Path, default=DEFAULT_NPY)
    p.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    args = p.parse_args()
    apply_labels(args.npy, args.zarr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
