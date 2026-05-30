"""
Per-sample frame images with π_high predicted option overlays burned in.

Saves one PNG per val/train sample into two subdirs on the Volume:
  correct/  — true_{skill}_{####}.png
  wrong/    — true_{skill}_pred_{skill}_{####}.png

Each image: 60px header (True/Pred skill names, CORRECT/WRONG) stacked above
the 3× upscaled 96×96 frame with the predicted skill border.

Usage:
    modal run modal/modal_analyze_high_images.py \\
        --checkpoint /experiments/pusht/train_high/soda_supervised/best.ckpt

    # download to local folder after saving:
    modal run modal/modal_analyze_high_images.py \\
        --checkpoint /experiments/pusht/train_high/soda_supervised/best.ckpt \\
        --local-dir C:/tmp/val_frames

    # train split:
    modal run modal/modal_analyze_high_images.py \\
        --checkpoint /experiments/pusht/train_high/soda_supervised/best.ckpt \\
        --split train --local-dir C:/tmp/train_frames

    # custom volume output dir:
    modal run modal/modal_analyze_high_images.py \\
        --checkpoint /experiments/pusht/train_high/soda_supervised/best.ckpt \\
        --output-dir /experiments/pusht/train_high/soda_supervised/my_frames
"""

import subprocess
import sys

from modal_config import analyze_high_images, app


@app.local_entrypoint()
def main(
    checkpoint: str,
    split: str = "val",
    max_images: int = 300,
    output_dir: str = "",
    scale: int = 3,
    local_dir: str = "",
) -> None:
    result = analyze_high_images.remote(
        checkpoint,
        split=split,
        max_images=max_images,
        output_dir=output_dir or None,
        scale=scale,
    )
    remote_dir = result["output_dir"]
    n_saved = result["n_saved"]
    n_wrong = result["n_wrong"]
    print(f"\nSaved {n_saved} images (wrong={n_wrong}) → Volume:{remote_dir}")

    if local_dir:
        print(f"\nDownloading to {local_dir} ...")
        ret = subprocess.run(
            ["modal", "volume", "get", "soda-experiments", remote_dir, local_dir],
            check=False,
        )
        if ret.returncode == 0:
            print(f"Done. Images in {local_dir}/")
        else:
            print("Download failed — run manually:", file=sys.stderr)
            print(f"  modal volume get soda-experiments {remote_dir} {local_dir}", file=sys.stderr)
    else:
        print(f"\nTo download locally:")
        print(f"  modal volume get soda-experiments {remote_dir} .")
