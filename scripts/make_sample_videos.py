"""
Generate 2 labeled sample videos from block_push episodes for visual review.
Saves: sample_ep0.mp4, sample_ep1.mp4 in the current directory.

Run from repo root:
  conda activate soda
  python scripts/make_sample_videos.py
"""
import os, glob
import numpy as np
import cv2

DATA_DIR   = "data/raw/block_push"
OUT_DIR    = "data/raw/block_push"
EPISODES   = [0, 1]
FPS        = 10

OPTION_NAMES  = ["reach_first", "push_first", "reach_second", "push_second"]
OPTION_COLORS = {
    "reach_first":  (100, 180, 255),
    "push_first":   ( 30,  80, 220),
    "reach_second": (255, 160,  60),
    "push_second":  (200,  80,   0),
}

def burn(frame, idx, option_name):
    out = frame.copy()
    color = OPTION_COLORS[option_name]
    h, w = out.shape[:2]
    # colored border
    cv2.rectangle(out, (0, 0), (w-1, h-1), color, 4)
    # frame number (red, top-left)
    cv2.putText(out, str(idx), (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 1, cv2.LINE_AA)
    # option name (colored, bottom-left)
    cv2.putText(out, option_name, (6, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out

files = sorted(glob.glob(os.path.join(DATA_DIR, "episode_*.npz")))
print(f"Found {len(files)} episodes in {DATA_DIR}")

for ep_idx in EPISODES:
    d = np.load(files[ep_idx])
    images     = d["images"]     # (T, H, W, 3)
    option_ids = d["option_id"]  # (T,)
    T = len(images)

    out_path = os.path.join(OUT_DIR, f"sample_ep{ep_idx}.mp4")
    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    for t in range(T):
        frame = burn(images[t], t, OPTION_NAMES[option_ids[t]])
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()

    # Print segment summary
    changes = [0] + [i for i in range(1, T) if option_ids[i] != option_ids[i-1]] + [T]
    segs = [(changes[i], changes[i+1], OPTION_NAMES[option_ids[changes[i]]]) for i in range(len(changes)-1)]
    print(f"\nEpisode {ep_idx} ({T} frames) → {out_path}")
    for s, e, name in segs:
        print(f"  frames {s:3d}-{e:3d}  {name}")

print("\nDone. Open sample_ep0.mp4 and sample_ep1.mp4 to review labels.")
