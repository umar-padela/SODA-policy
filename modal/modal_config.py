"""
Shared Modal infrastructure for SODA-policy.

Imported by modal_smoke.py, modal_train_low.py, modal_train_high.py, modal_eval.py,
modal_rollout_low_policy.py.

Layout
------
- ``app``       — groups this project's Modal functions (dashboard name).
- ``image``     — Linux + CUDA Python environment (mirrors environment.modal.yml / DP).
- ``volume``    — persistent disk for checkpoints & logs (survives after GPU stops).
- ``smoke``     — row-5 health check: torch, zarr, pusht.zarr on GPU.
- ``eval_run`` — generic Push-T eval (policy + rollouts + multi-horizon metrics).
- ``rollout_low_policy`` / ``list_rollout_segments`` — expert-anchored π_low segment rollout.
- ``train_low`` / ``train_high`` — remote training on ``GPU_TRAIN`` (default L40S).

Local vs remote
---------------
Your laptop only runs ``modal run ...`` (see modal_train_*.py).
Functions decorated with ``@app.function`` run inside ``image`` on Modal's GPUs.

Entrypoints use ``.spawn()`` (not ``.remote()``) so jobs keep running if the local
``modal run`` client disconnects; use ``--detach`` on train scripts to return
immediately without waiting on ``FunctionCall.get()``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# App & paths
# ---------------------------------------------------------------------------

app = modal.App("soda-policy")

# Repo root inside the container (set by add_local_dir below).
REPO_ROOT = "/root/soda-policy"
ZARR_PATH = f"{REPO_ROOT}/data/raw/pusht/pusht.zarr"
from soda.experiments.paths import (
    frozen_dp_pusht_checkpoint,
    hydra_run_dir,
    infer_task_slug,
    segment_rollout_dir,
    train_dp_dir,
    train_high_dir,
    train_low_dir,
    VOLUME_MOUNT as EXPERIMENTS_MOUNT,
)

# BID repo path inside the container.
BID_DIR = f"{REPO_ROOT}/third_party/bid_diffusion"
# Volume path for BID weak policy checkpoint (downloaded once via modal_download_bid.py).
BID_PUSHT_WEAK_CKPT = f"{EXPERIMENTS_MOUNT}/bid_checkpoints/pusht/epoch_0050.ckpt"

# Back-compat aliases (prefer paths module).
DP_FROZEN_VOLUME_DIR = str(frozen_dp_pusht_checkpoint(in_volume=True))
DP_BASELINE_VOLUME_DIR = DP_FROZEN_VOLUME_DIR
FROZEN_DP_PUSHT_CHECKPOINT = str(frozen_dp_pusht_checkpoint(in_volume=True))

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
        # Runtime libs for PyAV manylinux wheels (eval MP4 recording)
        "ffmpeg",
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
        "gym==0.23.1",  # 0.21 sdist fails on modern pip; use runner_common.vector_env_reset()
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
        "pybullet",       # block push data generation
    )
    # PyAV: conda pins 10.0.0; pip has no py3.10 wheel for 10.x (Cython build fails on Modal).
    # 12.3.0 ships manylinux binaries — same API used by diffusion_policy VideoRecorder.
    .pip_install("av==12.3.0")
    # Columbia frozen DP (hybrid workspace) needs robomimic + Columbia robosuite fork.
    # That fork pins numba<=0.53.1 (no py3.10 wheels) — install with --no-deps; numba 0.56.4 above.
    .pip_install("cffi==1.15.1")
    .run_commands(
        "pip install --no-cache-dir 'mujoco-py==2.1.2.14'",
        'pip install --no-cache-dir --no-deps "robosuite @ https://github.com/cheng-chi/robosuite/archive/277ab9588ad7a4f4b55cf75508b44aa67ec171f0.tar.gz"',
        "pip install --no-cache-dir 'robomimic==0.2.0'",
    )
    .run_commands(
        # Pre-bake ResNet18 ImageNet weights so containers don't download on every cold start.
        "python -c \"import torchvision; torchvision.models.resnet18(pretrained=True)\"",
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

# Modal GPU shortcodes — override locally without editing code:
#   set MODAL_GPU_EVAL=L4 && set MODAL_GPU_TRAIN=A100-40GB  (Windows)
#   MODAL_GPU_EVAL=L4 MODAL_GPU_TRAIN=A100-40GB modal run ...
GPU_EVAL = os.environ.get("MODAL_GPU_EVAL", "A10G")
GPU_TRAIN = os.environ.get("MODAL_GPU_TRAIN", "A100-40GB")
# Number of GPUs for training — set MODAL_NUM_GPUS=2 to request 2× A100-40GB + torchrun DDP.
NUM_GPUS_TRAIN = int(os.environ.get("MODAL_NUM_GPUS", "1"))
# Modal GPU spec: "A100-40GB" for 1 GPU, "A100-40GB:2" for 2 GPUs, etc.
_GPU_TRAIN_SPEC = f"{GPU_TRAIN}:{NUM_GPUS_TRAIN}" if NUM_GPUS_TRAIN > 1 else GPU_TRAIN

# W&B optional — attach only when logging. Create once:
#   modal secret create wandb WANDB_API_KEY=<key>
_wandb_secret = modal.Secret.from_name("wandb")


@app.function(
    image=image,
    gpu=None,
    timeout=600,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def download_bid_checkpoints(
    dest_dir: str = f"{EXPERIMENTS_MOUNT}/bid_checkpoints/pusht",
) -> str:
    """
    Download BID's pre-trained weak Push-T checkpoint (epoch_0050, score=0.250).

    Only the weak policy is needed — the strong policy is our existing Columbia DP best.ckpt.
    Source: https://github.com/YuejiangLIU/bid_diffusion (Google Drive folder).
    """
    import subprocess
    import shutil
    from pathlib import Path

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    weak_path = dest / "epoch_0050.ckpt"
    if weak_path.is_file():
        print(f"BID weak checkpoint already exists: {weak_path}")
        volume.commit()
        return str(weak_path)

    # Use gdown CLI (same as BID README: gdown --folder <url> -O <dest>)
    subprocess.run(["pip", "install", "gdown", "-q"], check=True)

    tmp_dir = Path("/tmp/bid_ckpts")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Correct folder ID from BID README
    folder_url = "https://drive.google.com/drive/folders/1o8rf2Lq91D_DCq7RqZVyFAP-eMcLOAP2"
    subprocess.run(
        ["gdown", "--folder", folder_url, "-O", str(tmp_dir)],
        check=True,
    )

    # Move .ckpt files into dest
    found = list(tmp_dir.rglob("*.ckpt"))
    if not found:
        raise FileNotFoundError(f"No .ckpt files found after gdown in {tmp_dir}")
    for f in found:
        target = dest / f.name
        shutil.move(str(f), str(target))
        print(f"  Saved: {target}")

    # Create canonical epoch_0050.ckpt symlink/copy if file has long name
    if not weak_path.exists():
        candidates = sorted(dest.glob("*0050*.ckpt"))
        if candidates:
            shutil.copy(str(candidates[0]), str(weak_path))
            print(f"  Canonical weak ckpt: {weak_path}")

    volume.commit()
    print(f"BID checkpoints ready in {dest}")
    return str(weak_path)


@app.function(
    image=image,
    gpu="A100-80GB",   # BID batches N=16 samples × n_envs; needs more VRAM than A10G
    timeout=14400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def eval_bid_pusht(
    *,
    checkpoint: str,
    reference: str,
    noise: float = 0.0,
    n_test: int = 25,
    n_samples: int = 16,
    n_mode: int = 3,
    decay: float = 0.9,
    output_dir: str,
    noise_rho: float = 0.9,
) -> dict:
    """
    Native BID evaluation on Push-T using our own Columbia DP image runner.

    BID's eval_bid.py subprocess approach doesn't work with the image-based DP
    (PushTImageRunner has no set_sampler). Instead, we implement BID natively:
    load strong + weak DP checkpoints, wrap with BIDPolicy, run via our runner.

    Returns per-episode scores compatible with the noise study JSON format.
    """
    import numpy as np
    from pathlib import Path

    from soda.eval.bid_sampler import BIDPolicy
    from soda.eval.dp_runner import build_pusht_image_runner
    from soda.eval.policy_loaders import load_dp_image_policy_and_cfg
    from soda.eval.runner_common import resolve_test_start_seed, serialize_runner_log
    from soda.training.dp_config import merge_eval_yaml_into_ckpt_cfg

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "media").mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    strong, _settings, ckpt_cfg = load_dp_image_policy_and_cfg(checkpoint, device=device)
    strong.n_action_steps = 1

    # Load weak policy only if reference is a compatible image-based checkpoint.
    # BID's provided checkpoints are lowdim/keypoints — incompatible with our image CNN.
    # Without a weak policy, BID runs positive-only forward contrast (still beats vanilla).
    weak = None
    if reference:
        try:
            from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
            weak, _, _ = load_dp_image_policy_and_cfg(reference, device=device)
            if not isinstance(weak, DiffusionUnetHybridImagePolicy):
                print(f"Weak policy is not an image policy ({type(weak).__name__}) — running BID without negative contrast")
                weak = None
            else:
                weak.n_action_steps = 1
                print(f"Loaded weak policy: {reference}")
        except Exception as e:
            print(f"Could not load weak policy ({e}) — running BID without negative contrast")
            weak = None

    bid_policy = BIDPolicy(strong, weak, n_samples=n_samples, n_mode=n_mode, decay=decay)

    seed = resolve_test_start_seed(100000, ckpt_cfg)
    runner = build_pusht_image_runner(
        ckpt_cfg,
        out_dir,
        n_test=n_test,
        n_test_vis=n_test,   # record all episodes (score in filename)
        n_action_steps=1,
        max_steps=300,
        test_start_seed=seed,
        noise_eta=noise,
        noise_rho=noise_rho,
    )

    runner_log = runner.run(bid_policy)
    soda_metrics = runner_log.pop("soda_metrics", {})

    per_episode_scores = [
        float(runner_log.get(f"test/sim_max_reward_{seed + i}", 0.0))
        for i in range(n_test)
    ]
    mean_score = float(np.mean(per_episode_scores)) if per_episode_scores else 0.0
    std_score = float(np.std(per_episode_scores)) if per_episode_scores else 0.0

    volume.commit()
    return {
        "policy": "bid",
        "noise_eta": float(noise),
        "noise_rho": noise_rho,
        "n_action_steps": 1,
        "n_episodes": len(per_episode_scores),
        "mean_score": mean_score,
        "std_score": std_score,
        "per_episode_scores": per_episode_scores,
        "checkpoint": str(checkpoint),
        "reference": str(reference),
        "metrics": soda_metrics,
    }


def _run(
    cmd: list[str],
    cwd: str | None = None,
    *,
    run_readme: str | None = None,
    invoke_command: str | None = None,
) -> None:
    from soda.experiments.run_readme import INVOKE_COMMAND_ENV, RUN_README_ENV, validate_run_readme

    env = os.environ.copy()
    env["HYDRA_FULL_ERROR"] = "1"
    if run_readme is not None:
        env[RUN_README_ENV] = validate_run_readme(run_readme)
    if invoke_command:
        env[INVOKE_COMMAND_ENV] = invoke_command
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True, env=env)


def _volume_train_low_dir(task: str, config_name: str) -> str:
    """Persistent checkpoint dir on Modal Volume (not ephemeral container FS)."""
    return str(train_low_dir(task, config_name, in_volume=True))


def _volume_train_high_dir(task: str, config_name: str) -> str:
    return str(train_high_dir(task, config_name, in_volume=True))


def _volume_train_dp_dir(task: str, config_name: str) -> str:
    return str(train_dp_dir(task, config_name, in_volume=True))


def _parse_hydra_overrides(hydra_overrides: str | list[str] | None) -> list[str]:
    """Split space-separated Hydra overrides from Modal CLI."""
    if hydra_overrides is None:
        return []
    if isinstance(hydra_overrides, list):
        return [o for o in hydra_overrides if o.strip()]
    return [part for part in hydra_overrides.split() if part.strip()]


def _override_key_set(overrides: list[str]) -> set[str]:
    return {o.split("=", 1)[0] for o in overrides if "=" in o}


def _merge_hydra_overrides(
    defaults: dict[str, str] | None,
    hydra_overrides: list[str] | None,
) -> list[str]:
    """Apply ``defaults`` unless the same Hydra key appears in ``hydra_overrides``."""
    keys = _override_key_set(hydra_overrides or [])
    merged = [f"{key}={value}" for key, value in (defaults or {}).items() if key not in keys]
    if hydra_overrides:
        merged.extend(hydra_overrides)
    return merged


def _build_train_cmd(
    script: str,
    *,
    task: str,
    config_name: str,
    hydra_subdir: str,
    default_overrides: dict[str, str] | None = None,
    hydra_overrides: list[str] | None = None,
    use_torchrun: bool = False,
) -> list[str]:
    hydra_dir = Path(hydra_run_dir(task, hydra_subdir, config_name, in_volume=True))
    hydra_dir.mkdir(parents=True, exist_ok=True)
    overrides = _merge_hydra_overrides(default_overrides, hydra_overrides)
    # Hydra resolves --config-path relative to the @hydra.main script (soda/training/),
    # not cwd — use an absolute repo path so Modal and local subprocess calls agree.
    config_dir = Path(REPO_ROOT) / "configs" / task
    # Use torchrun for multi-GPU DDP training; fall back to plain python for single GPU.
    # Detect GPU count at call time (inside container) rather than at import time (on laptop),
    # because MODAL_NUM_GPUS is not propagated into the container environment.
    if use_torchrun:
        try:
            import torch as _torch
            n_gpus = _torch.cuda.device_count() if _torch.cuda.is_available() else 1
        except Exception:
            n_gpus = NUM_GPUS_TRAIN  # fallback to locally-read value
        if n_gpus > 1:
            launcher = ["torchrun", f"--nproc_per_node={n_gpus}"]
        else:
            launcher = ["python"]
    else:
        launcher = ["python"]
    cmd = launcher + [
        script,
        "--config-path",
        str(config_dir),
        "--config-name",
        config_name,
        f"hydra.run.dir={hydra_dir}",
    ]
    cmd.extend(overrides)
    return cmd


def spawn_modal_function(
    modal_function: modal.Function,
    *,
    label: str,
    wait: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Start a Modal function with ``.spawn()`` (preferred over ``.remote()``).

    ``.spawn()`` creates an independent FunctionCall on Modal's servers.  The job
    survives even if the local process disconnects — but ONLY when the local
    entrypoint was invoked with ``modal run --detach`` (the CLI flag BEFORE the
    script path).  Without that flag, Modal stops the ephemeral app when the
    entrypoint exits and cancels any spawned jobs.

    ``wait=True``  → ``.spawn().get()``: blocks until the remote function returns
                      (live logs stream to your terminal).  Close the terminal
                      any time if you used ``--detach``; the job keeps running.
    ``wait=False`` → ``.spawn()`` only: entrypoint returns immediately.  Use this
                      when the entrypoint itself is not the top-level modal run
                      (e.g. the sequential sweep entrypoint).
    """
    call = modal_function.spawn(**kwargs)
    print(f"Spawned {label} (object_id={call.object_id})")
    if not wait:
        print("Detached — monitor in Modal dashboard; job continues if this CLI exits.")
        return call
    return call.get()


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_EVAL,
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
    gpu=GPU_EVAL,
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def download_frozen_dp(
    dest_path: str = FROZEN_DP_PUSHT_CHECKPOINT,
    best: bool = False,
) -> str:
    """
    One-time download of Columbia frozen DP checkpoint onto the Volume.

    best=False (default): downloads latest.ckpt (final training checkpoint).
    best=True:            downloads best.ckpt (best validation checkpoint).
    """
    from soda.eval.dp_frozen import (
        DP_PUSHT_IMAGE_TRAIN0_BEST_URL,
        ensure_dp_checkpoint,
    )

    if best:
        resolved_dest = dest_path.replace("latest.ckpt", "best.ckpt")
        url = DP_PUSHT_IMAGE_TRAIN0_BEST_URL
    else:
        resolved_dest = dest_path
        url = None

    path = ensure_dp_checkpoint(Path(resolved_dest), download=True, url=url)
    volume.commit()
    print(f"Frozen DP checkpoint ready: {path}")
    return str(path)


@app.function(
    image=image,
    gpu=GPU_EVAL,
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def eval_run(
    config_path: str = "configs/pusht/dp_frozen.yaml",
    *,
    run_readme: str,
    full: bool = False,
    checkpoint_path: str | None = None,
    high_checkpoint: str | None = None,
    low_checkpoint: str | None = None,
    n_test: int | None = None,
    max_steps: int | None = None,
    n_action_steps: int | None = None,
    n_test_vis: int | None = None,
    ckpt_slug: str | None = None,
    record_video: bool = True,
    invoke_command: str | None = None,
    output_dir: str | None = None,
    noise_eta: float = 0.0,
    noise_rho: float = 0.9,
    noise_scale_by_magnitude: bool = False,
) -> dict:
    """
    Generic Push-T eval on Modal: load yaml → roll out → overlap @ 125–300 (25-step grid).

    See ``modal/modal_eval.py`` and ``soda/eval/eval_yaml.py``.
    """
    from pathlib import Path

    from soda.experiments.run_readme import INVOKE_COMMAND_ENV, RUN_README_ENV, validate_run_readme
    from soda.eval.run_eval import (
        EvalCliOverrides,
        build_eval_config_from_yaml,
        resolve_eval_output_dir,
        run_pusht_eval,
    )

    os.environ[RUN_README_ENV] = validate_run_readme(run_readme)
    if invoke_command:
        os.environ[INVOKE_COMMAND_ENV] = invoke_command

    cli = EvalCliOverrides(
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        high_checkpoint=Path(high_checkpoint) if high_checkpoint else None,
        low_checkpoint=Path(low_checkpoint) if low_checkpoint else None,
        full=full,
        n_test=n_test,
        max_steps=max_steps,
        n_action_steps=n_action_steps,
        n_test_vis=n_test_vis,
        ckpt_slug=ckpt_slug,
        record_video=record_video,
        run_readme=run_readme,
        noise_eta=noise_eta,
        noise_rho=noise_rho,
        noise_scale_by_magnitude=noise_scale_by_magnitude,
    )
    cfg = build_eval_config_from_yaml(config_path, cli=cli)

    if output_dir is not None:
        from datetime import datetime, timezone
        out_dir = Path(EXPERIMENTS_MOUNT) / output_dir.lstrip("/").removeprefix(EXPERIMENTS_MOUNT.lstrip("/")).lstrip("/")
        run_ts = datetime.now(timezone.utc)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir, run_ts = resolve_eval_output_dir(cfg, Path(EXPERIMENTS_MOUNT))
    result = run_pusht_eval(
        cfg,
        out_dir,
        experiments_root=Path(EXPERIMENTS_MOUNT),
        invoke_command=invoke_command,
        cli=cli,
        run_timestamp=run_ts,
    )

    volume.commit()
    result["output_dir"] = str(out_dir)
    print("eval_run OK:", {k: result[k] for k in result if k != "metrics"})
    return result


def resolve_volume_path(path: str | Path, *, mount: str = EXPERIMENTS_MOUNT) -> Path:
    """Resolve checkpoint/output paths on the Modal Volume."""
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(mount) / p


def default_low_checkpoint_for_config(config_path: str) -> Path:
    """``/experiments/{task}/train_low/{config_stem}/best.ckpt`` from eval yaml path."""
    from soda.eval.eval_yaml import resolve_eval_config_path

    resolved = resolve_eval_config_path(config_path)
    task = infer_task_slug(config_path=resolved)
    stem = resolved.stem
    return Path(train_low_dir(task, stem, in_volume=True)) / "best.ckpt"


@app.function(
    image=image,
    timeout=600,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def list_rollout_segments(
    *,
    episode_idx: int | None = None,
    option_id: int | None = None,
    label_key: str = "option_id_supervised",
    zarr_path: str | None = None,
    list_limit: int = 30,
) -> dict:
    """Print matching zarr option segments (CPU; zarr baked into image)."""
    from soda.eval.segment_zarr import ZarrSegmentStore, list_segments_text

    store_path = zarr_path or ZARR_PATH
    store = ZarrSegmentStore.open(store_path, label_key=label_key)
    text = list_segments_text(
        store,
        option_id=option_id,
        episode_idx=episode_idx,
        limit=list_limit,
    )
    print(text)
    return {
        "zarr_path": str(store_path),
        "label_key": label_key,
        "total_segments": len(store.segments),
        "listing": text,
    }


@app.function(
    image=image,
    gpu=GPU_EVAL,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def rollout_low_policy(
    config_path: str = "configs/pusht/soda_supervised.yaml",
    *,
    checkpoint: str | None = None,
    segment_index: int | None = None,
    episode_idx: int | None = None,
    option_id: int | None = None,
    segment_rank: int = 0,
    fixed_option_id: int | None = None,
    label_key: str = "option_id_supervised",
    zarr_path: str | None = None,
    n_action_steps: int = 8,
    max_steps: int | None = None,
    output_dir: str | None = None,
    no_video: bool = False,
    representative_samples: bool = False,
    per_skill: int = 2,
    invoke_command: str | None = None,
) -> dict:
    """
    Expert-anchored π_low rollout on Modal (see ``scripts/rollout_segment.py``).

    Checkpoints and outputs live under ``/experiments`` on Volume ``soda-experiments``.
    Default checkpoint: ``/experiments/train_low/{config_stem}/best.ckpt``.
    With ``representative_samples=True``, runs ``per_skill`` rollouts per skill (default 6).
    """
    from soda.eval.eval_yaml import resolve_eval_config_path
    from soda.eval.segment_rollout import (
        run_representative_segment_rollouts,
        run_segment_rollout_from_cli,
    )

    resolved_config = resolve_eval_config_path(config_path)
    task = infer_task_slug(config_path=resolved_config)
    ckpt_path = (
        resolve_volume_path(checkpoint)
        if checkpoint is not None
        else default_low_checkpoint_for_config(config_path)
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"π_low checkpoint not found on Volume: {ckpt_path}\n"
            "Train with modal_train_low.py or pass --checkpoint explicitly."
        )

    out_dir = (
        resolve_volume_path(output_dir)
        if output_dir is not None
        else Path(segment_rollout_dir(task, in_volume=True))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if invoke_command:
        (out_dir / "invoke_command.txt").write_text(invoke_command + "\n", encoding="utf-8")

    if representative_samples:
        from soda.eval.segment_zarr import REPRESENTATIVE_MIN_SEGMENT_LENGTH
        results = run_representative_segment_rollouts(
            checkpoint=ckpt_path,
            config_path=resolved_config,
            zarr_path=Path(zarr_path or ZARR_PATH),
            per_skill=per_skill,
            min_segment_length=REPRESENTATIVE_MIN_SEGMENT_LENGTH,
            fixed_option_id=fixed_option_id,
            label_key=label_key,
            device="cuda:0",
            n_action_steps=n_action_steps,
            max_steps=max_steps,
            output_dir=out_dir,
            no_video=no_video,
        )
        volume.commit()
        rollouts = []
        for result in results:
            rollouts.append(
                {
                    "segment_index": result.segment_index,
                    "anchor_frame": result.anchor_frame,
                    "fixed_option_id": result.fixed_option_id,
                    "n_policy_steps": result.n_policy_steps,
                    "metrics": result.metrics,
                    "video_path": str(result.video_path) if result.video_path else None,
                }
            )
            print(
                "rollout_low_policy OK:",
                {
                    "segment_index": result.segment_index,
                    "skill": result.metrics.get("skill"),
                    "video_path": str(result.video_path) if result.video_path else None,
                },
            )
        payload = {
            "config_path": str(resolved_config),
            "checkpoint": str(ckpt_path),
            "output_dir": str(out_dir),
            "representative_samples": True,
            "per_skill": per_skill,
            "n_rollouts": len(rollouts),
            "rollouts": rollouts,
        }
        return payload

    result = run_segment_rollout_from_cli(
        checkpoint=ckpt_path,
        config_path=resolved_config,
        zarr_path=Path(zarr_path or ZARR_PATH),
        segment_index=segment_index,
        episode_idx=episode_idx,
        option_id=option_id,
        segment_rank=segment_rank,
        fixed_option_id=fixed_option_id,
        label_key=label_key,
        device="cuda:0",
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        output_dir=out_dir,
        no_video=no_video,
    )

    volume.commit()
    payload = {
        "config_path": str(resolved_config),
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
        "segment_index": result.segment_index,
        "anchor_frame": result.anchor_frame,
        "fixed_option_id": result.fixed_option_id,
        "n_policy_steps": result.n_policy_steps,
        "metrics": result.metrics,
        "video_path": str(result.video_path) if result.video_path else None,
    }
    print("rollout_low_policy OK:", {k: payload[k] for k in payload if k != "metrics"})
    return payload


@app.function(
    image=image,
    gpu=GPU_EVAL,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def rollout_dp_policy(
    *,
    checkpoint: str | None = None,
    zarr_path: str | None = None,
    label_key: str = "option_id_supervised",
    n_action_steps: int = 8,
    max_steps: int | None = None,
    output_dir: str | None = None,
    no_video: bool = False,
    per_skill: int = 2,
    invoke_command: str | None = None,
) -> dict:
    """
    Expert-anchored vanilla DP segment rollout on Modal.

    Resets sim to zarr segment start states and rolls out the Columbia frozen DP
    (no option conditioning, no β head). Use ``per_skill`` to control how many
    representative segments per skill are rolled out (default 2 → 6 total).

    Default checkpoint: /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt
    Outputs land under /experiments/pusht/segment_rollout/dp_baseline/ on the Volume.
    """
    from soda.eval.segment_rollout import run_representative_segment_rollouts_dp
    from soda.eval.segment_zarr import REPRESENTATIVE_MIN_SEGMENT_LENGTH

    ckpt_path = (
        resolve_volume_path(checkpoint)
        if checkpoint is not None
        else Path(FROZEN_DP_PUSHT_CHECKPOINT)
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"DP checkpoint not found on Volume: {ckpt_path}\n"
            "Download with: modal run modal/modal_download_dp.py"
        )

    out_dir = (
        resolve_volume_path(output_dir)
        if output_dir is not None
        else Path(segment_rollout_dir("pusht", in_volume=True)) / "dp_baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if invoke_command:
        (out_dir / "invoke_command.txt").write_text(invoke_command + "\n", encoding="utf-8")

    results = run_representative_segment_rollouts_dp(
        dp_checkpoint=ckpt_path,
        zarr_path=Path(zarr_path or ZARR_PATH),
        per_skill=per_skill,
        min_segment_length=REPRESENTATIVE_MIN_SEGMENT_LENGTH,
        label_key=label_key,
        device="cuda:0",
        n_action_steps=n_action_steps,
        max_steps=max_steps,
        output_dir=out_dir,
        record_video=not no_video,
    )
    volume.commit()

    rollouts = []
    for r in results:
        rollouts.append(
            {
                "segment_index": r.segment_index,
                "anchor_frame": r.anchor_frame,
                "fixed_option_id": r.fixed_option_id,
                "n_policy_steps": r.n_policy_steps,
                "metrics": r.metrics,
                "video_path": str(r.video_path) if r.video_path else None,
            }
        )
        print(
            "rollout_dp_policy OK:",
            {
                "segment_index": r.segment_index,
                "skill": r.metrics.get("skill"),
                "max_overlap": r.metrics.get("max_overlap_full"),
                "video_path": str(r.video_path) if r.video_path else None,
            },
        )

    return {
        "checkpoint": str(ckpt_path),
        "output_dir": str(out_dir),
        "per_skill": per_skill,
        "n_rollouts": len(rollouts),
        "rollouts": rollouts,
    }


def default_high_checkpoint_for_config(config_path: str) -> Path:
    """``/experiments/{task}/train_high/{config_stem}/best.ckpt`` from eval yaml path."""
    from soda.eval.eval_yaml import resolve_eval_config_path

    resolved = resolve_eval_config_path(config_path)
    task = infer_task_slug(config_path=resolved)
    stem = resolved.stem
    return Path(train_high_dir(task, stem, in_volume=True)) / "best.ckpt"


def hierarchical_rollout_dir(task: str = "pusht") -> str:
    """Volume path for hierarchical rollout outputs."""
    return str(Path(EXPERIMENTS_MOUNT) / task / "hierarchical_rollout")


@app.function(
    image=image,
    gpu=GPU_EVAL,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def rollout_hierarchical(
    config_path: str = "configs/pusht/soda_supervised.yaml",
    *,
    high_checkpoint: str | None = None,
    low_checkpoint: str | None = None,
    weak_low_checkpoint: str | None = None,
    n_episodes: int = 5,
    test_start_seed: int = 100000,
    n_action_steps: int = 8,
    beta_transition: float | None = None,
    open_loop: bool = False,
    duration_termination: bool = False,
    high_monitors_every_step: bool = False,
    max_steps: int = 300,
    output_dir: str | None = None,
    no_video: bool = False,
    video_failure_threshold: float | None = 20.0,
    noise_eta: float = 0.0,
    noise_rho: float = 0.9,
    noise_scale_by_magnitude: bool = False,
    invoke_command: str | None = None,
) -> dict:
    """
    Full-episode hierarchical rollout (π_high + π_low + β) on Modal.

    ``video_failure_threshold``: save video only for episodes where max_overlap_full
    is below this percentage (default 20.0). Set to None to save all videos when
    no_video=False, or ignored entirely when no_video=True.

    Produces annotated MP4s (frame idx, skill border, β, duration, cursor, REPLAN)
    and per-episode JSON summaries under ``/experiments/{task}/hierarchical_rollout/``.

    ``open_loop=True``: π_high resamples ω after every full native chunk; β disabled.
    Default checkpoints: best.ckpt from train_high / train_low for the config.
    """
    from soda.eval.eval_yaml import resolve_eval_config_path
    from soda.eval.hierarchical_rollout import run_hierarchical_rollouts

    resolved_config = resolve_eval_config_path(config_path)
    task = infer_task_slug(config_path=resolved_config)

    high_ckpt = (
        resolve_volume_path(high_checkpoint)
        if high_checkpoint is not None
        else default_high_checkpoint_for_config(config_path)
    )
    low_ckpt = (
        resolve_volume_path(low_checkpoint)
        if low_checkpoint is not None
        else default_low_checkpoint_for_config(config_path)
    )

    for label, path in (("π_high", high_ckpt), ("π_low", low_ckpt)):
        if not path.is_file():
            raise FileNotFoundError(
                f"{label} checkpoint not found on Volume: {path}\n"
                "Train first or pass --high-checkpoint / --low-checkpoint explicitly."
            )

    out_dir = (
        resolve_volume_path(output_dir)
        if output_dir is not None
        else Path(hierarchical_rollout_dir(task))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if invoke_command:
        (out_dir / "invoke_command.txt").write_text(invoke_command + "\n", encoding="utf-8")

    print(
        f"rollout_hierarchical: n_episodes={n_episodes} n_action_steps={n_action_steps} "
        f"beta_transition={beta_transition} max_steps={max_steps}"
    )
    print(f"  π_high: {high_ckpt}")
    print(f"  π_low:  {low_ckpt}")

    weak_low_ckpt = (
        resolve_volume_path(weak_low_checkpoint)
        if weak_low_checkpoint is not None
        else None
    )

    results = run_hierarchical_rollouts(
        config_path=resolved_config,
        high_checkpoint=high_ckpt,
        low_checkpoint=low_ckpt,
        weak_low_checkpoint=weak_low_ckpt,
        device="cuda:0",
        n_episodes=n_episodes,
        test_start_seed=test_start_seed,
        n_action_steps=n_action_steps,
        beta_transition=beta_transition,
        open_loop=open_loop,
        duration_termination=duration_termination,
        high_monitors_every_step=high_monitors_every_step,
        max_steps=max_steps,
        output_dir=out_dir,
        record_video=not no_video,
        video_overlap_threshold=None if no_video else video_failure_threshold,
        noise_eta=noise_eta,
        noise_rho=noise_rho,
        noise_scale_by_magnitude=noise_scale_by_magnitude,
    )

    volume.commit()

    episodes = []
    for r in results:
        episodes.append(
            {
                "episode_idx": r.episode_idx,
                "seed": r.seed,
                "n_steps": r.n_steps,
                "n_replans": r.n_replans,
                "metrics": r.metrics,
                "option_sequence": r.option_sequence[:20],  # truncate for JSON readability
                "video_path": str(r.video_path) if r.video_path else None,
            }
        )
        print(
            f"rollout_hierarchical ep={r.episode_idx}: "
            f"steps={r.n_steps} replans={r.n_replans} "
            f"max_overlap={r.metrics.get('max_overlap_full', 0):.1f}% "
            f"video={r.video_path}"
        )

    mean_score = float(
        sum(e["metrics"].get("max_overlap_full", 0.0) for e in episodes) / len(episodes)
    ) if episodes else 0.0

    payload = {
        "config_path": str(resolved_config),
        "high_checkpoint": str(high_ckpt),
        "low_checkpoint": str(low_ckpt),
        "output_dir": str(out_dir),
        "n_episodes": len(episodes),
        "mean_max_overlap": mean_score,
        "episodes": episodes,
    }
    print(f"rollout_hierarchical OK: n={len(episodes)} mean_max_overlap={mean_score:.1f}%")
    return payload


@app.function(
    image=image,
    gpu=_GPU_TRAIN_SPEC,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def run_sweep_train_low_sequential(
    trials: list[dict],
) -> None:
    """
    Run π_low sweep trials sequentially on a single GPU container.

    Each trial dict: {name, task, config_name, hydra_overrides (list[str]), run_readme}.
    All trials share one GPU — no concurrent quota issues; laptop can disconnect.
    """
    n = len(trials)
    for i, trial in enumerate(trials, 1):
        name = trial["name"]
        print(f"\n{'=' * 60}\n=== Sweep trial {i}/{n}: {name} ===\n{'=' * 60}")
        overrides = _parse_hydra_overrides(trial.get("hydra_overrides") or [])
        cmd = _build_train_cmd(
            "soda/training/train_low.py",
            task=trial["task"],
            config_name=trial["config_name"],
            hydra_subdir="sweep_low",
            default_overrides={},
            hydra_overrides=overrides,
        )
        _run(cmd, run_readme=trial.get("run_readme", ""))
        volume.commit()
        print(f"Trial {name} done.")
    print(f"\nAll {n} sweep trials complete.")


@app.function(
    image=image,
    gpu=_GPU_TRAIN_SPEC,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def run_sweep_train_high_sequential(
    trials: list[dict],
) -> None:
    """
    Run π_high sweep trials sequentially on a single GPU container.

    Each trial dict: {name, task, config_name, hydra_overrides (list[str]), run_readme}.
    """
    n = len(trials)
    for i, trial in enumerate(trials, 1):
        name = trial["name"]
        print(f"\n{'=' * 60}\n=== Sweep trial {i}/{n}: {name} ===\n{'=' * 60}")
        overrides = _parse_hydra_overrides(trial.get("hydra_overrides") or [])
        cmd = _build_train_cmd(
            "soda/training/train_high.py",
            task=trial["task"],
            config_name=trial["config_name"],
            hydra_subdir="sweep_high",
            default_overrides={},
            hydra_overrides=overrides,
        )
        _run(cmd, run_readme=trial.get("run_readme", ""))
        volume.commit()
        print(f"Trial {name} done.")
    print(f"\nAll {n} sweep trials complete.")


@app.function(
    image=image,
    gpu=_GPU_TRAIN_SPEC,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train_dp(
    hydra_overrides: list[str],
    *,
    run_readme: str,
    invoke_command: str | None = None,
) -> None:
    """
    Train vanilla DP using Columbia's pipeline (train.py) directly.
    All config is passed as Hydra overrides — no wrapper config translation.
    Checkpoints go to wherever hydra.run.dir points.
    """
    cmd = ["python", "soda/training/train_dp_direct.py"] + list(hydra_overrides)
    _run(cmd, run_readme=run_readme, invoke_command=invoke_command)
    volume.commit()


@app.function(
    image=image,
    gpu=_GPU_TRAIN_SPEC,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train_low(
    config_name: str = "soda_supervised",
    task: str = "pusht",
    hydra_overrides: list[str] | None = None,
    *,
    run_readme: str,
    invoke_command: str | None = None,
) -> None:
    """
    Remote wrapper around soda/training/train_low.py (Hydra / DP-style CLI).

    Checkpoints are written to ``/experiments/train_low/{config_name}/`` on the Volume
    (override with ``train_low.output_dir=...`` in ``hydra_overrides``).
    Set MODAL_NUM_GPUS=N before ``modal run`` to use N-GPU DDP (torchrun).
    """
    overrides = _parse_hydra_overrides(hydra_overrides)
    cmd = _build_train_cmd(
        "soda/training/train_low.py",
        task=task,
        config_name=config_name,
        hydra_subdir="train_low",
        default_overrides={
            "train_low.output_dir": _volume_train_low_dir(task, config_name),
        },
        hydra_overrides=overrides,
        use_torchrun=True,
    )
    _run(cmd, run_readme=run_readme, invoke_command=invoke_command)
    volume.commit()


@app.function(
    image=image,
    gpu=_GPU_TRAIN_SPEC,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train_beta(
    config_name: str = "obs_positive_negative_standalone",
    task: str = "pusht",
    hydra_overrides: list[str] | None = None,
    *,
    run_readme: str,
    invoke_command: str | None = None,
) -> None:
    """
    Remote wrapper around soda/training/train_beta.py (standalone β network).

    Trains a fresh ResNet (ImageNet init) + option embed + MLP as a standalone
    termination classifier, decoupled from the LowPolicy diffusion backbone.
    Checkpoints are written to the path specified in train_beta.output_dir
    (set via hydra_overrides or the yaml config).
    """
    overrides = _parse_hydra_overrides(hydra_overrides)
    cmd = _build_train_cmd(
        "soda/training/train_beta.py",
        task=task,
        config_name=config_name,
        hydra_subdir="train_beta",
        hydra_overrides=overrides,
        use_torchrun=False,  # single-GPU only; no DDP needed for standalone beta
    )
    _run(cmd, run_readme=run_readme, invoke_command=invoke_command)
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_EVAL,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def rollout_hierarchical_external_beta(
    config_path: str,
    *,
    high_checkpoint: str,
    low_checkpoint: str,
    standalone_beta_checkpoint: str,
    n_episodes: int = 50,
    test_start_seed: int = 100000,
    n_action_steps: int = 8,
    beta_transition: float | None = None,
    max_steps: int = 300,
    output_dir: str | None = None,
    no_video: bool = False,
    video_failure_threshold: float | None = 20.0,
    invoke_command: str | None = None,
) -> dict:
    """
    Full-episode hierarchical rollout using a StandaloneBeta as the termination signal.

    π_low (k=9 checkpoint) generates actions; StandaloneBeta decides when to switch options.
    Mirrors rollout_hierarchical but accepts a separate standalone_beta_checkpoint.
    """
    from pathlib import Path

    from soda.eval.hierarchical_rollout import run_hierarchical_rollouts
    from soda.eval.policy_loaders import load_standalone_beta_from_checkpoint
    from soda.experiments.run_readme import INVOKE_COMMAND_ENV

    import numpy as np

    if invoke_command:
        os.environ[INVOKE_COMMAND_ENV] = invoke_command

    resolved_config = Path(REPO_ROOT) / config_path
    high_ckpt = resolve_volume_path(high_checkpoint)
    low_ckpt = resolve_volume_path(low_checkpoint)
    beta_ckpt = resolve_volume_path(standalone_beta_checkpoint)

    for label, path in (("π_high", high_ckpt), ("π_low", low_ckpt), ("β", beta_ckpt)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found on Volume: {path}")

    out_dir = (
        resolve_volume_path(output_dir)
        if output_dir is not None
        else Path(hierarchical_rollout_dir()) / "stage4_standalone_beta"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if invoke_command:
        (out_dir / "invoke_command.txt").write_text(invoke_command + "\n", encoding="utf-8")

    external_beta = load_standalone_beta_from_checkpoint(str(beta_ckpt), device="cuda:0")

    results = run_hierarchical_rollouts(
        config_path=resolved_config,
        high_checkpoint=high_ckpt,
        low_checkpoint=low_ckpt,
        device="cuda:0",
        n_episodes=n_episodes,
        test_start_seed=test_start_seed,
        n_action_steps=n_action_steps,
        beta_transition=beta_transition,
        open_loop=False,
        duration_termination=False,
        max_steps=max_steps,
        output_dir=out_dir,
        record_video=not no_video,
        video_overlap_threshold=None if no_video else video_failure_threshold,
        external_beta=external_beta,
    )

    volume.commit()

    episodes = []
    for r in results:
        episodes.append({
            "episode_idx": r.episode_idx,
            "seed": r.seed,
            "n_steps": r.n_steps,
            "n_replans": r.n_replans,
            "metrics": r.metrics,
            "option_sequence": r.option_sequence[:20],
            "video_path": str(r.video_path) if r.video_path else None,
        })

    mean_score = float(
        sum(e["metrics"].get("max_overlap_full", 0.0) for e in episodes) / len(episodes)
    ) if episodes else 0.0

    payload = {
        "config_path": str(resolved_config),
        "high_checkpoint": str(high_ckpt),
        "low_checkpoint": str(low_ckpt),
        "standalone_beta_checkpoint": str(beta_ckpt),
        "output_dir": str(out_dir),
        "n_episodes": len(episodes),
        "mean_max_overlap": mean_score,
        "episodes": episodes,
    }
    print(f"rollout_hierarchical_external_beta OK: n={len(episodes)} mean_max_overlap={mean_score:.1f}%")
    return payload


@app.function(
    image=image,
    gpu=_GPU_TRAIN_SPEC,
    timeout=28800,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train_high(
    config_name: str = "soda_supervised",
    task: str = "pusht",
    low_checkpoint: str | None = None,
    hydra_overrides: list[str] | None = None,
    *,
    run_readme: str,
    invoke_command: str | None = None,
) -> None:
    """
    Remote wrapper around soda/training/train_high.py.

    π_high needs a trained π_low checkpoint (frozen vision encoder). Pass
    ``low_checkpoint`` or ``train_high.low_checkpoint=...`` in ``hydra_overrides``.
    Checkpoints go to ``/experiments/train_high/{config_name}/`` on the Volume.
    """
    overrides = _parse_hydra_overrides(hydra_overrides)
    if low_checkpoint:
        overrides = _merge_hydra_overrides(
            {"train_high.low_checkpoint": low_checkpoint},
            overrides,
        )
    # Respect output_dir set in the yaml config; only fall back to the default
    # volume path if the config leaves it null.
    import yaml as _yaml
    _config_yaml = Path(REPO_ROOT) / "configs" / task / f"{config_name}.yaml"
    _yaml_output_dir = None
    if _config_yaml.exists():
        with open(_config_yaml) as _f:
            _y = _yaml.safe_load(_f) or {}
        _yaml_output_dir = (_y.get("train_high") or {}).get("output_dir")
    output_dir = _yaml_output_dir or _volume_train_high_dir(task, config_name)
    cmd = _build_train_cmd(
        "soda/training/train_high.py",
        task=task,
        config_name=config_name,
        hydra_subdir="train_high",
        default_overrides={
            "train_high.output_dir": output_dir,
        },
        hydra_overrides=overrides,
        use_torchrun=True,
    )
    _run(cmd, run_readme=run_readme, invoke_command=invoke_command)
    volume.commit()


@app.function(
    image=image,
    gpu=None,
    timeout=600,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def analyze_high(checkpoint: str, split: str = "val") -> dict:
    """Confusion matrix + per-class accuracy for a trained π_high checkpoint."""
    import dill
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from soda.eval.policy_loaders import load_high_policy_from_checkpoint
    from soda.training.train_high import build_datasets

    print(f"Loading π_high from {checkpoint} ...")
    policy = load_high_policy_from_checkpoint(checkpoint, device="cpu")
    policy.eval()

    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
    saved_cfg = payload.get("cfg")
    if saved_cfg is None:
        raise ValueError("Checkpoint has no saved cfg — cannot rebuild dataset.")
    cfg = OmegaConf.create(saved_cfg) if isinstance(saved_cfg, dict) else saved_cfg
    OmegaConf.update(cfg, "task.dataset.zarr_path", ZARR_PATH, merge=True)

    train_ds, val_ds = build_datasets(cfg)
    ds = val_ds if split == "val" else train_ds
    print(f"Dataset split={split}: {len(ds)} samples")

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    num_options = policy.cfg.num_options
    confusion = np.zeros((num_options, num_options), dtype=int)

    with torch.no_grad():
        for batch in loader:
            labels = batch["option_id"].reshape(-1).long()
            feat = policy.encode_obs(batch["obs"])
            prev_opt_id = batch.get("prev_option_id", None)
            preds = policy.sample_option(feat, prev_option_id=prev_opt_id)
            for true, pred in zip(labels.numpy(), preds.cpu().numpy()):
                confusion[true, pred] += 1

    print(f"\nConfusion matrix  ({split})  — rows=true label, cols=predicted")
    print(f"{'':>10}", end="")
    for j in range(num_options):
        print(f"  pred_{j:>2}", end="")
    print("   | recall  (n)")
    for i in range(num_options):
        print(f"  true_{i:>3} ", end="")
        for j in range(num_options):
            marker = "*" if i == j else " "
            print(f" {confusion[i, j]:>5}{marker}", end="")
        n = int(confusion[i].sum())
        recall = confusion[i, i] / n if n > 0 else 0.0
        print(f"   | {recall:.1%}  ({n})")

    print("\nPer-class precision:")
    for j in range(num_options):
        n_pred = int(confusion[:, j].sum())
        prec = confusion[j, j] / n_pred if n_pred > 0 else 0.0
        print(f"  option_{j}: {prec:.1%}  ({confusion[j, j]}/{n_pred} predicted as {j})")

    total_correct = int(np.diag(confusion).sum())
    total = int(confusion.sum())
    overall = total_correct / total if total > 0 else 0.0
    print(f"\nOverall {split} accuracy: {overall:.1%}  ({total_correct}/{total})")

    errors = [
        (confusion[i, j], i, j)
        for i in range(num_options)
        for j in range(num_options)
        if i != j and confusion[i, j] > 0
    ]
    print("\nTop misclassifications (true → predicted, count):")
    for count, true, pred in sorted(errors, reverse=True):
        print(f"  option_{true} → option_{pred}: {count}")

    return {"confusion": confusion.tolist(), "overall_acc": overall, "split": split}


@app.function(
    image=image,
    gpu=None,
    timeout=600,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def analyze_high_images(
    checkpoint: str,
    split: str = "val",
    max_images: int = 300,
    output_dir: str | None = None,
    scale: int = 3,
) -> dict:
    """
    Save one annotated PNG per val sample; organise into correct/ and wrong/ subdirs.

    Each image: 60px header (true/pred skill names, CORRECT/WRONG status) stacked above
    the upscaled (96×scale px) frame with the predicted option border burned in.

    ``output_dir`` is a path on the Modal Volume (default: ``{ckpt_dir}/{split}_frames/``).
    Returns ``{"output_dir": str, "n_saved": int, "n_wrong": int}``.
    """
    import cv2
    import dill
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from soda.eval.policy_loaders import load_high_policy_from_checkpoint
    from soda.option_discovery.supervised.pusht.frame_overlays import (
        SKILL_COLORS_BGR,
        burn_option_overlay,
    )
    from soda.option_discovery.supervised.pusht.heuristics import SKILL_NAMES
    from soda.training.train_high import build_datasets

    FRAME_SZ = 96
    HEADER_H = 60

    print(f"Loading π_high from {checkpoint} ...")
    policy = load_high_policy_from_checkpoint(checkpoint, device="cpu")
    policy.eval()

    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
    saved_cfg = payload.get("cfg")
    if saved_cfg is None:
        raise ValueError("Checkpoint has no saved cfg — cannot rebuild dataset.")
    cfg = OmegaConf.create(saved_cfg) if isinstance(saved_cfg, dict) else saved_cfg
    OmegaConf.update(cfg, "task.dataset.zarr_path", ZARR_PATH, merge=True)

    train_ds, val_ds = build_datasets(cfg)
    ds = val_ds if split == "val" else train_ds
    print(f"Dataset split={split}: {len(ds)} samples")

    ckpt_dir = str(Path(checkpoint).parent)
    out_dir = Path(output_dir or f"{ckpt_dir}/{split}_frames")
    (out_dir / "correct").mkdir(parents=True, exist_ok=True)
    (out_dir / "wrong").mkdir(parents=True, exist_ok=True)

    n_wrong = 0
    counters: dict[str, int] = {}
    W = FRAME_SZ * scale

    with torch.no_grad():
        for idx in range(min(len(ds), max_images)):
            sample = ds[idx]
            obs = {k: v.unsqueeze(0) for k, v in sample["obs"].items()}
            true_id = int(sample["option_id"].item())
            prev_opt_tensor = sample.get("prev_option_id", None)
            if prev_opt_tensor is not None:
                prev_opt_tensor = prev_opt_tensor.reshape(1)
            feat = policy.encode_obs(obs)
            pred_id = int(policy.sample_option(feat, prev_option_id=prev_opt_tensor).item())
            correct = true_id == pred_id
            if not correct:
                n_wrong += 1

            # float32 CHW [0,1] (last obs step) → uint8 HWC BGR
            img_chw = sample["obs"]["image"][-1].numpy()  # (3, 96, 96)
            img_bgr = cv2.cvtColor(
                (img_chw.transpose(1, 2, 0) * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )

            # Burn predicted option border + skill name on 96×96, then upscale
            frame = burn_option_overlay(img_bgr, pred_id)
            frame_up = cv2.resize(frame, (W, W), interpolation=cv2.INTER_NEAREST)

            # Header bar: skill-color bg (red if wrong), true+pred text
            true_name = SKILL_NAMES.get(true_id, str(true_id))
            pred_name = SKILL_NAMES.get(pred_id, str(pred_id))
            bg_color = (0, 0, 180) if not correct else SKILL_COLORS_BGR.get(pred_id, (100, 100, 100))
            header = np.full((HEADER_H, W, 3), bg_color, dtype=np.uint8)
            status_text = "CORRECT" if correct else "WRONG"
            status_color = (0, 200, 0) if correct else (0, 0, 255)
            cv2.putText(header, f"True:  {true_name}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(header, f"Pred:  {pred_name}", (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(header, status_text, (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)

            img_out = np.vstack([header, frame_up])

            # Filename: true skill name + counter, so files group naturally by skill
            key = f"{'correct' if correct else 'wrong'}_{true_name}"
            counters[key] = counters.get(key, 0) + 1
            subdir = "correct" if correct else "wrong"
            if correct:
                fname = f"true_{true_name}_{counters[key]:04d}.png"
            else:
                fname = f"true_{true_name}_pred_{pred_name}_{counters[key]:04d}.png"
            cv2.imwrite(str(out_dir / subdir / fname), img_out)

    volume.commit()
    n_saved = min(len(ds), max_images)
    print(f"Saved {n_saved} images → {out_dir}/  (wrong={n_wrong}/{n_saved})")
    print(f"  correct/ — {n_saved - n_wrong} images")
    print(f"  wrong/   — {n_wrong} images")
    return {"output_dir": str(out_dir), "n_saved": n_saved, "n_wrong": n_wrong}


@app.function(
    image=image,
    gpu="T4",
    timeout=14400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train_love() -> None:
    """Remote wrapper around the LOVE adapter trainer (E3 unsupervised, Push-T).

    Writes the best checkpoint and the action k-means centroids to the
    persistent volume under /experiments/love_pusht/.
    """
    out_dir = Path(EXPERIMENTS_MOUNT) / "love_pusht"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SODA_LOVE_CKPT_DIR"] = str(out_dir)
    subprocess.run(
        [
            "python",
            "-m",
            "soda.option_discovery.unsupervised.love_adapter.train",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    volume.commit()


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def label_love() -> None:
    """Run the trained LOVE checkpoint over pusht.zarr; write labels.npy to volume.

    Reads ckpt from /experiments/love_pusht/best.ckpt; reads zarr from the
    read-only image; writes labels to
    /experiments/love_pusht/option_id_unsupervised.npy.

    Pull locally with:
        modal volume get soda-experiments love_pusht/option_id_unsupervised.npy ./experiments/love_pusht/
    Then apply via:
        python scripts/apply_love_labels.py
    """
    out_dir = Path(EXPERIMENTS_MOUNT) / "love_pusht"
    ckpt = out_dir / "best.ckpt"
    npy_out = out_dir / "option_id_unsupervised.npy"
    subprocess.run(
        [
            "python",
            "-m",
            "soda.option_discovery.unsupervised.love_adapter.label",
            "--ckpt",
            str(ckpt),
            "--output-npy",
            str(npy_out),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    volume.commit()
