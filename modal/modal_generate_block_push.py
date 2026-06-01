"""
Generate 1000 block push expert demonstrations on Modal and build the zarr.

Data is saved to the soda-experiments volume:
  /experiments/data/raw/block_push/episode_XXXX.npz   (raw episodes)
  /experiments/data/raw/block_push/block_push.zarr     (training-ready zarr)

Usage:
  modal run --detach modal/modal_generate_block_push.py

Estimated runtime: ~2 hours (CPU, 1000 episodes × ~7s each).
Re-running is safe — skips episodes that already exist on the volume.
"""
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).parent))
from modal_config import app, image, volume, EXPERIMENTS_MOUNT  # noqa: E402

# Extend the base image with pybullet (not needed for training, only data gen).
block_push_image = image.pip_install("pybullet")

VOLUME_DATA_DIR  = f"{EXPERIMENTS_MOUNT}/data/raw/block_push"
VOLUME_ZARR_PATH = f"{VOLUME_DATA_DIR}/block_push.zarr"
REPO_ROOT        = "/root/soda-policy"
N_EPISODES       = 1000


@app.function(
    image=block_push_image,
    volumes={EXPERIMENTS_MOUNT: volume},
    cpu=4,
    timeout=3 * 3600,   # 3 hours to be safe
)
def generate_and_build():
    import subprocess, os

    # ── How many episodes already exist? ─────────────────────────────────────
    existing = list(Path(VOLUME_DATA_DIR).glob("episode_*.npz")) if Path(VOLUME_DATA_DIR).exists() else []
    n_existing = len(existing)
    print(f"Volume has {n_existing}/{N_EPISODES} episodes already.")

    if n_existing < N_EPISODES:
        print(f"Generating episodes {n_existing} → {N_EPISODES - 1} ...")
        result = subprocess.run(
            [
                "python",
                f"{REPO_ROOT}/soda/option_discovery/supervised/block_push/generate_data.py",
                "--n-episodes", str(N_EPISODES),
                "--out-dir",    VOLUME_DATA_DIR,
            ],
            cwd=REPO_ROOT,
            capture_output=False,
        )
        if result.returncode != 0:
            raise RuntimeError("generate_data.py failed")
        volume.commit()
        print("Episodes saved to volume.")
    else:
        print("All episodes present — skipping generation.")

    # ── Build zarr ────────────────────────────────────────────────────────────
    print(f"\nBuilding zarr at {VOLUME_ZARR_PATH} ...")
    result = subprocess.run(
        [
            "python",
            f"{REPO_ROOT}/soda/option_discovery/supervised/block_push/build_zarr.py",
            "--npz-dir",  VOLUME_DATA_DIR,
            "--zarr-out", VOLUME_ZARR_PATH,
        ],
        cwd=REPO_ROOT,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("build_zarr.py failed")

    volume.commit()
    print(f"\nDone. Zarr at {VOLUME_ZARR_PATH}")


@app.local_entrypoint()
def main():
    print(f"Launching block push generation ({N_EPISODES} episodes) on Modal...")
    print(f"Output: {VOLUME_ZARR_PATH}")
    generate_and_build.remote()
    print("Job complete.")
