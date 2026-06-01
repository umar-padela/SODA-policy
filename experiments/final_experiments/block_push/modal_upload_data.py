"""
Upload block_push.zarr to Modal volume so training can access it.

Run once after generating + converting data locally:
  modal run experiments/final_experiments/block_push/modal_upload_data.py

Uploads: data/raw/block_push/block_push.zarr
     To: /experiments/data/raw/block_push/block_push.zarr (on soda-experiments volume)
"""
import sys
from pathlib import Path
import modal

sys.path.insert(0, str(Path(__file__).parents[3] / "modal"))
from modal_config import app, volume, EXPERIMENTS_MOUNT  # noqa: E402


@app.local_entrypoint()
def main():
    local_zarr = Path("data/raw/block_push/block_push.zarr")
    remote_base = f"{EXPERIMENTS_MOUNT}/data/raw/block_push"

    if not local_zarr.exists():
        print(f"ERROR: {local_zarr} not found.")
        print("Run first:")
        print("  python soda/option_discovery/supervised/block_push/generate_data.py --n-episodes 200")
        print("  python soda/option_discovery/supervised/block_push/build_zarr.py")
        return

    print(f"Uploading {local_zarr} → {remote_base}/block_push.zarr ...")
    # Walk the zarr directory and upload all files
    count = 0
    for local_path in sorted(local_zarr.rglob("*")):
        if local_path.is_file():
            rel = local_path.relative_to(local_zarr.parent)
            remote_path = f"{remote_base}/{rel}"
            volume.put_file(str(local_path), remote_path)
            count += 1

    volume.commit()
    print(f"Uploaded {count} files. Done.")
    print(f"\nTo verify:  modal volume ls soda-experiments data/raw/block_push/")
