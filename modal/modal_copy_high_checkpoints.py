"""
One-time script: copy high policy checkpoints into high_study/ on the Modal Volume.

  modal run modal/modal_copy_high_checkpoints.py
"""

import modal

app            = modal.App("soda-policy")
volume         = modal.Volume.from_name("soda-experiments", create_if_missing=False)
EXPERIMENTS_MOUNT = "/experiments"
_slim_image    = modal.Image.debian_slim(python_version="3.10")


@app.function(
    image=_slim_image,
    volumes={EXPERIMENTS_MOUNT: volume},
    timeout=300,
)
def copy_checkpoints() -> None:
    import shutil
    from pathlib import Path

    root = Path(EXPERIMENTS_MOUNT)

    moves = [
        (
            root / "final_experiments/pusht/high_starts_prev_opt",
            root / "final_experiments/pusht/high_study/high_starts_prev_opt",
        ),
        (
            root / "final_experiments/pusht/high_all_frames",
            root / "final_experiments/pusht/high_study/high_all_frames",
        ),
    ]

    for src, dst in moves:
        if not src.exists():
            print(f"  SKIP (not found): {src}")
            continue
        if dst.exists():
            print(f"  SKIP (already exists): {dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        print(f"  Copied: {src} -> {dst}")

    volume.commit()
    print("Done.")


@app.local_entrypoint()
def main() -> None:
    copy_checkpoints.remote()
