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
EXPERIMENTS_MOUNT = "/experiments"
DP_FROZEN_VOLUME_DIR = f"{EXPERIMENTS_MOUNT}/dp_baselines/pusht_image_cnn_train0"
# Back-compat alias (volume path unchanged on Modal).
DP_BASELINE_VOLUME_DIR = DP_FROZEN_VOLUME_DIR
FROZEN_DP_PUSHT_CHECKPOINT = f"{DP_FROZEN_VOLUME_DIR}/latest.ckpt"

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
GPU_TRAIN = os.environ.get("MODAL_GPU_TRAIN", "L40S")

# W&B optional — attach only when logging. Create once:
#   modal secret create wandb WANDB_API_KEY=<key>
_wandb_secret = modal.Secret.from_name("wandb")


def _run(
    cmd: list[str],
    cwd: str | None = None,
    *,
    run_readme: str | None = None,
    invoke_command: str | None = None,
) -> None:
    from soda.experiments.run_readme import INVOKE_COMMAND_ENV, RUN_README_ENV, validate_run_readme

    env = os.environ.copy()
    if run_readme is not None:
        env[RUN_README_ENV] = validate_run_readme(run_readme)
    if invoke_command:
        env[INVOKE_COMMAND_ENV] = invoke_command
    subprocess.run(cmd, cwd=cwd or REPO_ROOT, check=True, env=env)


def _volume_train_low_dir(config_name: str) -> str:
    """Persistent checkpoint dir on Modal Volume (not ephemeral container FS)."""
    return f"{EXPERIMENTS_MOUNT}/train_low/{config_name}"


def _volume_train_high_dir(config_name: str) -> str:
    return f"{EXPERIMENTS_MOUNT}/train_high/{config_name}"


def _volume_train_dp_dir(config_name: str) -> str:
    return f"{EXPERIMENTS_MOUNT}/train_dp/{config_name}"


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
    defaults: dict[str, str],
    hydra_overrides: list[str] | None,
) -> list[str]:
    """Apply ``defaults`` unless the same Hydra key appears in ``hydra_overrides``."""
    keys = _override_key_set(hydra_overrides or [])
    merged = [f"{key}={value}" for key, value in defaults.items() if key not in keys]
    if hydra_overrides:
        merged.extend(hydra_overrides)
    return merged


def _build_train_cmd(
    script: str,
    *,
    task: str,
    config_name: str,
    hydra_subdir: str,
    default_overrides: dict[str, str],
    hydra_overrides: list[str] | None = None,
) -> list[str]:
    hydra_dir = Path(EXPERIMENTS_MOUNT) / hydra_subdir / f"{task}_{config_name}"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    overrides = _merge_hydra_overrides(default_overrides, hydra_overrides)
    # Hydra resolves --config-path relative to the @hydra.main script (soda/training/),
    # not cwd — use an absolute repo path so Modal and local subprocess calls agree.
    config_dir = Path(REPO_ROOT) / "configs" / task
    cmd = [
        "python",
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

    ``spawn`` returns a ``FunctionCall`` handle immediately, so detached / long
    jobs are not canceled when the local ``modal run`` process disconnects.
    When ``wait=True``, blocks on ``FunctionCall.get()`` for the result.
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
) -> str:
    """
    One-time download of Columbia frozen DP ``latest.ckpt`` onto the Volume.

    Eval does not download automatically — run this, then pass ``dest_path``
    to ``modal_eval.py --checkpoint``.
    """
    from soda.eval.dp_frozen import ensure_dp_checkpoint

    path = ensure_dp_checkpoint(Path(dest_path), download=True)
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
    )
    cfg = build_eval_config_from_yaml(config_path, cli=cli)

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
    """``/experiments/train_low/{config_stem}/best.ckpt`` from eval yaml path."""
    from soda.eval.eval_yaml import resolve_eval_config_path

    stem = resolve_eval_config_path(config_path).stem
    return Path(EXPERIMENTS_MOUNT) / "train_low" / stem / "best.ckpt"


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
    timeout=7200,
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
        else Path(EXPERIMENTS_MOUNT) / "segment_rollout"
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
    gpu=GPU_TRAIN,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
    secrets=[_wandb_secret],
)
def train_dp(
    config_name: str = "dp",
    task: str = "pusht",
    hydra_overrides: list[str] | None = None,
    *,
    run_readme: str,
    invoke_command: str | None = None,
) -> None:
    """
    Remote wrapper around soda/training/train_dp.py (Columbia DP workspace).

    Checkpoints are written to ``/experiments/train_dp/{config_name}/`` on the Volume
    (override with ``train_dp.output_dir=...`` in ``hydra_overrides``).
    """
    overrides = _parse_hydra_overrides(hydra_overrides)
    cmd = _build_train_cmd(
        "soda/training/train_dp.py",
        task=task,
        config_name=config_name,
        hydra_subdir="train_dp",
        default_overrides={
            "train_dp.output_dir": _volume_train_dp_dir(config_name),
            "checkpoint.volume_path": f"{_volume_train_dp_dir(config_name)}/latest.ckpt",
        },
        hydra_overrides=overrides,
    )
    _run(cmd, run_readme=run_readme, invoke_command=invoke_command)
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_TRAIN,
    timeout=14400,
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
    """
    overrides = _parse_hydra_overrides(hydra_overrides)
    cmd = _build_train_cmd(
        "soda/training/train_low.py",
        task=task,
        config_name=config_name,
        hydra_subdir="train_low",
        default_overrides={
            "train_low.output_dir": _volume_train_low_dir(config_name),
        },
        hydra_overrides=overrides,
    )
    _run(cmd, run_readme=run_readme, invoke_command=invoke_command)
    volume.commit()


@app.function(
    image=image,
    gpu=GPU_TRAIN,
    timeout=14400,
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
    cmd = _build_train_cmd(
        "soda/training/train_high.py",
        task=task,
        config_name=config_name,
        hydra_subdir="train_high",
        default_overrides={
            "train_high.output_dir": _volume_train_high_dir(config_name),
        },
        hydra_overrides=overrides,
    )
    _run(cmd, run_readme=run_readme, invoke_command=invoke_command)
    volume.commit()
