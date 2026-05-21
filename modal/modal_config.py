"""
Shared Modal infrastructure for SODA-policy.

Imported by modal_smoke.py, modal_train_low.py, modal_train_high.py, modal_eval.py.

Layout
------
- ``app``       — groups this project's Modal functions (dashboard name).
- ``image``     — Linux + CUDA Python environment (mirrors environment.modal.yml / DP).
- ``volume``    — persistent disk for checkpoints & logs (survives after GPU stops).
- ``smoke``     — row-5 health check: torch, zarr, pusht.zarr on GPU.
- ``eval_run`` — generic Push-T eval (policy + rollouts + multi-horizon metrics).
- ``train_low`` — runs soda/training/train_low.py on a remote GPU (when implemented).

Local vs remote
---------------
Your laptop only runs ``modal run ...`` (see modal_train_*.py).
Functions decorated with ``@app.function`` run inside ``image`` on Modal's GPUs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App & paths
# ---------------------------------------------------------------------------

app = modal.App("soda-policy")

# Repo root inside the container (set by add_local_dir below).
REPO_ROOT = "/root/soda-policy"
ZARR_PATH = f"{REPO_ROOT}/data/raw/pusht/pusht.zarr"
EXPERIMENTS_MOUNT = "/experiments"
DP_BASELINE_VOLUME_DIR = f"{EXPERIMENTS_MOUNT}/dp_baselines/pusht_image_cnn_train0"
# Columbia frozen Push-T image CNN (train_0). Download once via ``download_frozen_dp``.
FROZEN_DP_PUSHT_CHECKPOINT = f"{DP_BASELINE_VOLUME_DIR}/latest.ckpt"

# ---------------------------------------------------------------------------
# Image = recipe for the remote machine (OS packages + Python deps + your code)
# ---------------------------------------------------------------------------
# apt: system libraries — see third_party/diffusion_policy/README.md (sim) + build tools.
# run_commands: download mujoco 2.1.0 (needed for robosuite / Square eval later).
# pip: pins aligned with environment.modal.yml / DP conda_environment.yaml (PyTorch 1.12, zarr 2.x).
# add_local_dir: copy project into the image LAST (Modal caches layers above this).

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        # Diffusion Policy sim (Ubuntu 20.04 list)
        "libosmesa6-dev",
        "libgl1-mesa-glx",
        "libglfw3",
        "patchelf",
        # Headless rendering / GL (assignments + Square path)
        "libegl1",
        "libglew-dev",
        # Build C/Cython wheels (gym, pymunk, imagecodecs, ...)
        "g++",
        "make",
        "cmake",
        "swig",
        "git",
        "curl",
        "wget",
        "libglib2.0-0",
    )
    .run_commands(
        "mkdir -p /root/.mujoco",
        "wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz",
        "tar -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco",
        "rm /tmp/mujoco210.tar.gz",
    )
    .env(
        {
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "MUJOCO_PY_MUJOCO_PATH": "/root/.mujoco/mujoco210",
            "LD_LIBRARY_PATH": "/root/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu",
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT}/third_party/diffusion_policy",
        }
    )
    .pip_install(
        "torch==1.12.1",
        "torchvision==0.13.1",
        extra_index_url="https://download.pytorch.org/whl/cu116",
    )
    .pip_install("pip<24.1")   
    .pip_install(
        "numpy==1.23.3",
        "zarr==2.12.0",
        "numcodecs==0.10.2",
        "h5py==3.7.0",
        "hydra-core==1.2.0",
        "omegaconf==2.2.3",
        "einops==0.4.1",
        "tqdm==4.64.1",
        "dill==0.3.5.1",
        "pandas==1.5.3",
        "matplotlib==3.6.1",
        "scikit-image==0.19.3",
        "imageio==2.22.0",
        "imageio-ffmpeg==0.4.7",
        "Cython==0.29.32",
        "gym==0.23.1",
        "pymunk==6.2.1",
        "opencv-python-headless==4.6.0.66",
        "wandb==0.13.3",
        "diffusers==0.11.1",
        "huggingface_hub==0.16.4",
        "accelerate==0.13.2",
        "termcolor==2.0.1",
        "pygame==2.1.2",
        "scipy==1.9.1",
        "numba==0.56.4",
        "shapely==1.8.4",
        "psutil==5.9.2",
        "click==8.0.4",
    )
    # Columbia frozen DP (hybrid workspace) needs robomimic + Columbia robosuite fork.
    # That fork pins numba<=0.53.1 (no py3.10 wheels) — install with --no-deps; numba 0.56.4 above.
    .pip_install("cffi==1.15.1")
    .run_commands(
        "pip install --no-cache-dir 'mujoco-py==2.1.2.14'",
        'pip install --no-cache-dir --no-deps "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"',
        "pip install --no-cache-dir 'robomimic==0.2.0'",
    )
    .add_local_dir(
        ".",
        remote_path=REPO_ROOT,
        ignore=[
            "experiments/**",
            "assignment1/**",
            "assignment2/**",
            "assignment3/**",
            ".git/**",
            "**/__pycache__/**",
            "**/.pytest_cache/**",
            "data/processed/**",
        ],
    )
)

# ---------------------------------------------------------------------------
# Volume = cloud disk for training outputs (checkpoints, hydra logs, eval dumps)
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("soda-experiments", create_if_missing=True)

# W&B optional — attach only when logging. Create once:
#   modal secret create wandb WANDB_API_KEY=<key>
_wandb_secret = modal.Secret.from_name("wandb")


def _run(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True, env=os.environ.copy())


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="T4",
    timeout=600,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def smoke() -> dict:
    """
    Row-5 smoke test: GPU + torch + read pusht.zarr + W&B ping.

    Requires Modal secret ``wandb`` (``modal secret create wandb WANDB_API_KEY=...``).
    Logs to W&B project ``soda-debug`` only (not training runs).

    Returns plain JSON-serializable types only (no torch imports needed on laptop).
    """
    from datetime import datetime, timezone

    import torch
    import wandb
    import zarr

    store = zarr.open(ZARR_PATH, mode="r")
    option_labels = store["data/option_id_supervised"]
    option_len = int(option_labels.shape[0])
    cuda_available = bool(torch.cuda.is_available())
    cuda_device = str(torch.cuda.get_device_name(0)) if cuda_available else None

    run_name = f"modal-smoke-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project="soda-debug", job_type="smoke", name=run_name)
    wandb.log({"option_id_len": option_len, "cuda_available": int(cuda_available)})
    wandb.finish()

    out = {
        "torch": str(torch.__version__),
        "zarr": str(zarr.__version__),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "option_id_supervised_shape": list(option_labels.shape),
        "option_id_supervised_dtype": str(option_labels.dtype),
        "zarr_path": ZARR_PATH,
    }

    marker = Path(EXPERIMENTS_MOUNT) / "smoke_ok.txt"
    marker.write_text(
        f"torch={out['torch']} zarr={out['zarr']} shape={tuple(out['option_id_supervised_shape'])}\n",
        encoding="utf-8",
    )
    volume.commit()

    print("SODA Modal smoke OK:", out)
    return out


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def download_frozen_dp(
    dest_path: str = FROZEN_DP_PUSHT_CHECKPOINT,
) -> str:
    """
    One-time download of Columbia frozen DP ``latest.ckpt`` onto the Volume.

    Eval does not download automatically — run this, then pass ``dest_path``
    to ``modal_eval.py --checkpoint``.
    """
    from soda.eval.dp_baseline import ensure_dp_checkpoint

    path = ensure_dp_checkpoint(Path(dest_path), download=True)
    volume.commit()
    print(f"Frozen DP checkpoint ready: {path}")
    return str(path)


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def eval_run(
    checkpoint_path: str,
    policy_source: str = "dp_baseline",
    task: str = "pusht",
    soda_config_name: str = "soda_supervised",
    ckpt_slug: str | None = None,
    smoke: bool = True,
    regime: str = "receding",
    action_horizon: int = 1,
    n_test: int | None = None,
    max_steps: int = 300,
) -> dict:
    """
    Generic Push-T eval on Modal: load policy → roll out → overlap @ 150/200/250/300.

    See ``soda.eval.run_eval.EvalConfig`` and ``modal/modal_eval.py`` CLI.
    """
    from soda.eval.run_eval import EvalConfig, resolve_eval_output_dir, run_pusht_eval

    if n_test is None:
        n_test = 5 if smoke else 50

    ckpt = Path(checkpoint_path)
    cfg = EvalConfig(
        checkpoint_path=ckpt,
        task=task,  # type: ignore[arg-type]
        policy_source=policy_source,  # type: ignore[arg-type]
        soda_config_name=soda_config_name,
        ckpt_slug=ckpt_slug,
        device="cuda:0",
        regime=regime,  # type: ignore[arg-type]
        action_horizon=action_horizon,
        n_test=n_test,
        n_train=0,
        max_steps=max_steps,
    )

    out_dir = resolve_eval_output_dir(
        cfg, Path(EXPERIMENTS_MOUNT), ckpt_slug=ckpt_slug
    )
    result = run_pusht_eval(cfg, out_dir)

    volume.commit()
    result["output_dir"] = str(out_dir)
    print("eval_run OK:", {k: result[k] for k in result if k != "metrics"})
    return result


@app.function(
    image=image,
    gpu="T4",
    timeout=14400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train_low(config_name: str = "soda_supervised", task: str = "pusht") -> None:
    """
    Remote wrapper around soda/training/train_low.py (Hydra / DP-style CLI).

    Requires Modal secret ``wandb`` only if train_low enables W&B logging.
    """
    hydra_dir = Path(EXPERIMENTS_MOUNT) / "train_low" / f"{task}_{config_name}"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python",
            "soda/training/train_low.py",
            "--config-path",
            f"configs/{task}",
            "--config-name",
            config_name,
            f"hydra.run.dir={hydra_dir}",
        ],
    )
    volume.commit()


@app.function(
    image=image,
    gpu="T4",
    timeout=14400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train_high(config_name: str = "soda_supervised", task: str = "pusht") -> None:
    """Remote wrapper around soda/training/train_high.py."""
    hydra_dir = Path(EXPERIMENTS_MOUNT) / "train_high" / f"{task}_{config_name}"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python",
            "soda/training/train_high.py",
            "--config-path",
            f"configs/{task}",
            "--config-name",
            config_name,
            f"hydra.run.dir={hydra_dir}",
        ],
    )
    volume.commit()
