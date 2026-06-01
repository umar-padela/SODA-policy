"""
Convert generated block_push npz episodes → zarr dataset.

Zarr schema matches Push-T exactly so all existing training/dataset code works:
  data/img                  (N, 96, 96, 3)  uint8
  data/state                (N, 16)         float32
  data/action               (N, 2)          float32
  data/option_id_supervised (N,)            int32
  meta/episode_ends         (N_eps,)        int32

Images are center-cropped to square then resized to 96×96.

Usage (from repo root):
  python soda/option_discovery/supervised/block_push/build_zarr.py
  python soda/option_discovery/supervised/block_push/build_zarr.py \\
      --npz-dir data/raw/block_push \\
      --zarr-out data/raw/block_push/block_push.zarr
"""
import argparse, os, glob
import numpy as np
import cv2
import zarr
from tqdm import tqdm


IMG_SIZE = 96  # matches Push-T resolution


def center_crop_and_resize(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Center-crop to square then resize to size×size."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    cropped = img[y0:y0+s, x0:x0+s]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def build(npz_dir: str, zarr_out: str) -> None:
    files = sorted(glob.glob(os.path.join(npz_dir, "episode_*.npz")))
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found in {npz_dir}")
    print(f"Found {len(files)} episodes in {npz_dir}")

    all_imgs, all_states, all_actions, all_options = [], [], [], []
    episode_ends = []
    frame_count = 0

    for f in tqdm(files, desc="Loading episodes"):
        d = np.load(f)
        imgs    = d["images"]    # (T, H, W, 3)
        states  = d["state"]     # (T, 16)
        actions = d["actions"]   # (T, 2)
        options = d["option_id"] # (T,)

        T = len(imgs)
        resized = np.stack([center_crop_and_resize(imgs[t]) for t in range(T)], axis=0)

        all_imgs.append(resized)
        all_states.append(states)
        all_actions.append(actions)
        all_options.append(options)
        frame_count += T
        episode_ends.append(frame_count)

    print(f"Total frames: {frame_count} across {len(files)} episodes")

    all_imgs    = np.concatenate(all_imgs,    axis=0).astype(np.uint8)
    all_states  = np.concatenate(all_states,  axis=0).astype(np.float32)
    all_actions = np.concatenate(all_actions, axis=0).astype(np.float32)
    all_options = np.concatenate(all_options, axis=0).astype(np.int32)
    episode_ends = np.array(episode_ends, dtype=np.int32)

    print(f"img shape:    {all_imgs.shape}")
    print(f"state shape:  {all_states.shape}")
    print(f"action shape: {all_actions.shape}")
    print(f"options:      {np.unique(all_options)}")

    # Write zarr (overwrite if exists)
    if os.path.exists(zarr_out):
        import shutil
        shutil.rmtree(zarr_out)

    store = zarr.DirectoryStore(zarr_out)
    root  = zarr.group(store=store, overwrite=True)
    data  = root.create_group("data")
    meta  = root.create_group("meta")

    data.array("img",                  all_imgs,    chunks=(100, IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    data.array("state",                all_states,  chunks=(1000, all_states.shape[1]),  dtype="float32")
    data.array("action",               all_actions, chunks=(1000, 2),                    dtype="float32")
    data.array("option_id_supervised", all_options, chunks=(1000,),                      dtype="int32")
    meta.array("episode_ends",         episode_ends, dtype="int32")

    print(f"\nZarr written to: {zarr_out}")
    print(zarr.open(zarr_out).tree())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz-dir",  default="data/raw/block_push")
    p.add_argument("--zarr-out", default="data/raw/block_push/block_push.zarr")
    args = p.parse_args()
    build(args.npz_dir, args.zarr_out)
