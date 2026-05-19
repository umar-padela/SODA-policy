#!/usr/bin/env python3
"""
Build ``data/raw/pusht/pusht.zarr`` with heuristic ``option_id_supervised`` labels.

Colab notebook (visualization / threshold sanity checks):
https://colab.research.google.com/github/umar-padela/SODA-policy/blob/dev_umar/data/pusht/pushT_labeling.ipynb

Reads the Columbia Diffusion Policy Push-T replay zip (you download it once),
labels skills from state, and writes a DP-compatible directory store (Zarr v2).

If ``pusht.zarr`` already exists, only ``option_id_supervised`` is recomputed in place
(other arrays, including ``option_id_unsupervised``, are kept). Use ``--force`` for a
full rebuild from the Columbia zip.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES, label_skills

# Columbia DP Push-T replay (Google Drive — download manually, see README).
GDRIVE_URL = (
    "https://drive.google.com/uc?id=1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
)
DEFAULT_ZIP_NAME = "pusht_cchi_v7_replay.zarr.zip"

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ZIP_PATH = REPO_ROOT / "data" / "processed" / "pusht_cache" / DEFAULT_ZIP_NAME
DEFAULT_OUTPUT_ZARR = REPO_ROOT / "data" / "raw" / "pusht" / "pusht.zarr"


def _require_zarr_v2() -> None:
    if not zarr.__version__.startswith("2."):
        raise RuntimeError(
            f"Zarr v2 required (pip install 'zarr<3'). Found {zarr.__version__}"
        )


def require_columbia_zip(zip_path: Path) -> Path:
    """Ensure the Columbia replay zip is on disk."""
    if zip_path.is_file():
        return zip_path
    raise FileNotFoundError(
        f"Missing Columbia Push-T zip: {zip_path}\n\n"
        "Download once from Google Drive and save with this exact name:\n"
        f"  {GDRIVE_URL}\n"
        f"  → {zip_path}\n\n"
        "Then run:\n"
        "  python -m soda.option_discovery.supervised.pusht.build_zarr --force"
    )


def load_flat_dataset(zip_path: Path) -> dict[str, np.ndarray]:
    """Load Columbia zip into flat numpy arrays (rotation in degrees for labeling)."""
    store = zarr.storage.ZipStore(str(zip_path), mode="r")
    root = zarr.open(store=store, mode="r")
    dataset = {
        "images": np.asarray(root["data"]["img"][:]),
        "state": np.asarray(root["data"]["state"][:], dtype=np.float32),
        "action": np.asarray(root["data"]["action"][:], dtype=np.float32),
        "episode_ends": np.asarray(root["meta"]["episode_ends"][:], dtype=np.int32),
    }
    dataset["state"][:, 4] = np.rad2deg(dataset["state"][:, 4])
    print(
        f"Loaded {dataset['state'].shape[0]} frames, "
        f"{len(dataset['episode_ends'])} episodes"
    )
    return dataset


def _print_label_distribution(labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u} ({SKILL_NAMES.get(int(u), '?')}): {c}")


def _labeling_payload_from_state(
    state: np.ndarray, episode_ends: np.ndarray
) -> dict[str, np.ndarray]:
    """Build dict for ``label_skills`` (block rotation in degrees)."""
    state_deg = np.asarray(state, dtype=np.float32).copy()
    state_deg[:, 4] = np.rad2deg(state_deg[:, 4])
    return {"state": state_deg, "episode_ends": episode_ends}


def update_supervised_labels(output_dir: Path) -> None:
    """Recompute ``option_id_supervised`` from existing zarr state (no full rebuild)."""
    _require_zarr_v2()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Zarr not found: {output_dir}")

    root = zarr.open(str(output_dir), mode="a")
    state = np.asarray(root["data"]["state"][:], dtype=np.float32)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int32)

    labels = label_skills(_labeling_payload_from_state(state, episode_ends))
    _print_label_distribution(labels)

    data = root["data"]
    if "option_id_supervised" in data:
        data["option_id_supervised"][:] = labels
        print(f"Updated data/option_id_supervised in {output_dir}")
    else:
        data.array(
            "option_id_supervised",
            labels,
            chunks=(1000,),
            dtype="int32",
        )
        print(f"Created data/option_id_supervised in {output_dir}")


def write_pusht_zarr(dataset: dict[str, np.ndarray], output_dir: Path) -> None:
    """Write full DP-format zarr from Columbia flat arrays (replaces existing store)."""
    _require_zarr_v2()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    state = dataset["state"].copy()
    state[:, 4] = np.deg2rad(state[:, 4])

    labels = label_skills(dataset)
    _print_label_distribution(labels)

    store = zarr.DirectoryStore(str(output_dir))
    root = zarr.group(store=store, overwrite=True)
    data = root.create_group("data")
    meta = root.create_group("meta")

    data.array("img", dataset["images"], chunks=(100, 96, 96, 3), dtype="uint8")
    data.array("state", state, chunks=(1000, 5), dtype="float32")
    data.array("action", dataset["action"], chunks=(1000, 2), dtype="float32")
    data.array(
        "option_id_supervised",
        labels,
        chunks=(1000,),
        dtype="int32",
    )
    meta.array("episode_ends", dataset["episode_ends"], dtype="int32")

    print(f"Wrote zarr: {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help=f"Columbia replay zip (default: {DEFAULT_ZIP_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=DEFAULT_OUTPUT_ZARR,
        help="Output directory store path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild entire zarr from Columbia zip (default: update labels only if zarr exists)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.output_zarr.is_dir() and not args.force:
        print(f"Zarr exists — updating option_id_supervised only: {args.output_zarr}")
        update_supervised_labels(args.output_zarr)
        return 0

    try:
        require_columbia_zip(args.zip_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    dataset = load_flat_dataset(args.zip_path)
    write_pusht_zarr(dataset, args.output_zarr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
