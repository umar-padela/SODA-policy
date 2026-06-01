"""
Train π_low (lowdim) on Modal for block push.

  modal run --detach modal/modal_train_lowdim.py \\
      --run-readme "block_push lowdim k9 pi_low"
"""
import sys
from pathlib import Path

import modal

for _p in [str(Path(__file__).parent), "/root/soda-policy/modal"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from modal_config import app, image, volume, EXPERIMENTS_MOUNT, GPU_TRAIN  # noqa: E402

REPO_ROOT    = "/root/soda-policy"
TASK         = "block_push"
CONFIG_NAME  = "soda_lowdim"
ZARR_VOLUME  = f"{EXPERIMENTS_MOUNT}/data/raw/block_push/block_push.zarr"
OUTPUT_BASE  = f"{EXPERIMENTS_MOUNT}/train_low"


@app.function(
    image=image,
    gpu=GPU_TRAIN,
    volumes={EXPERIMENTS_MOUNT: volume},
    timeout=24 * 3600,
    secrets=[modal.Secret.from_name("wandb", required=False)],
)
def train_lowdim(
    config_name: str = CONFIG_NAME,
    task: str = TASK,
    hydra_overrides: list[str] | None = None,
    run_readme: str = "",
) -> None:
    import subprocess, os

    out_dir = f"{OUTPUT_BASE}/{task}_{config_name}"
    overrides = hydra_overrides or []
    # Always point at volume data
    overrides = [f"task.dataset.zarr_path={ZARR_VOLUME}",
                 f"train_low.output_dir={out_dir}"] + overrides

    override_str = " ".join(f'"{o}"' for o in overrides)
    cmd = (
        f"python {REPO_ROOT}/soda/training/train_low_lowdim.py "
        f"--config-path {REPO_ROOT}/configs/{task} "
        f"--config-name {config_name} "
        f"{override_str}"
    )
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=REPO_ROOT)
    volume.commit()
    if result.returncode != 0:
        raise RuntimeError("Training failed")


@app.local_entrypoint()
def main(
    run_readme: str = "block_push lowdim k9 pi_low",
    config_name: str = CONFIG_NAME,
    hydra_overrides: str = "",
) -> None:
    overrides = [p for p in hydra_overrides.split() if p.strip()] if hydra_overrides else None
    print(f"Launching lowdim π_low training on Modal — {TASK}/{config_name}")
    train_lowdim.remote(
        config_name     = config_name,
        task            = TASK,
        hydra_overrides = overrides,
        run_readme      = run_readme,
    )
    print("Job submitted.")
