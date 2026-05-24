"""
Frozen Columbia Diffusion Policy Push-T — checkpoint download helpers.

Eval rollouts live in ``run_eval.py`` + ``dp_runner.py``. Use config
``configs/pusht/dp_frozen.yaml`` with ``eval.policy: dp_frozen``.

For self-trained vanilla DP, use ``configs/pusht/dp.yaml`` with ``eval.policy: dp``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Official Columbia weights — image CNN, seed train_0 (not low_dim).
DP_PUSHT_IMAGE_TRAIN0_BASE = (
    "https://diffusion-policy.cs.columbia.edu/data/experiments/"
    "image/pusht/diffusion_policy_cnn/train_0/checkpoints"
)
DP_PUSHT_IMAGE_TRAIN0_LATEST_URL = f"{DP_PUSHT_IMAGE_TRAIN0_BASE}/latest.ckpt"

# Canonical path on Modal Volume (see ``modal download_frozen_dp``).
FROZEN_DP_PUSHT_CKPT_FILENAME = "latest.ckpt"
DEFAULT_FROZEN_DP_VOLUME_PATH = Path(
    "/experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt"
)


def ensure_dp_checkpoint(
    dest_path: Path,
    *,
    download: bool = True,
) -> Path:
    """Return path to frozen Push-T image checkpoint, downloading if needed."""
    dest_path = Path(dest_path)
    if dest_path.is_file() and dest_path.stat().st_size > 1_000_000:
        return dest_path

    if not download:
        raise FileNotFoundError(f"Checkpoint not found: {dest_path}")

    url = DP_PUSHT_IMAGE_TRAIN0_LATEST_URL
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DP checkpoint (~3 GB) from:\n  {url}\n  -> {dest_path}")
    subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(dest_path), url],
        check=True,
    )
    if not dest_path.is_file():
        raise RuntimeError(f"Download failed: {dest_path}")
    return dest_path


def run_dp_frozen_pusht_eval(
    checkpoint: str | Path,
    output_dir: str | Path,
    device: str = "cuda:0",
    *,
    smoke: bool = False,
) -> dict:
    """Back-compat wrapper around :func:`run_pusht_eval` for frozen DP."""
    from soda.eval.run_eval import EvalConfig, run_pusht_eval

    cfg = EvalConfig(
        config_path=Path("configs/pusht/dp_frozen.yaml"),
        config_name="dp_frozen",
        policy_source="dp_frozen",
        checkpoint_path=Path(checkpoint),
        device=device,
        n_test=5 if smoke else 50,
        n_train=0,
        show_progress=True,
    )
    result = run_pusht_eval(cfg, Path(output_dir))
    return result["metrics"]


# Legacy alias (deprecated name).
run_dp_pusht_eval = run_dp_frozen_pusht_eval
