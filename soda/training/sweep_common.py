"""
Shared helpers for short hyperparameter sweeps (training_plan.md A4 / B4).

Runs ``train_low.py`` / ``train_high.py`` via subprocess, then ranks trials from
``metrics.json`` using **best-epoch** val metrics (same criterion as ``best.ckpt``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_VOLUME_ROOT = "/experiments"
MODAL_VOLUME_NAME = "soda-experiments"

_OUTPUT_DIR_KEYS = ("train_low.output_dir", "train_high.output_dir")
_VOLUME_PATH_KEYS = (*_OUTPUT_DIR_KEYS, "train_high.low_checkpoint")


@dataclass(frozen=True)
class SweepTrial:
    """One sweep run: Hydra overrides plus a short label for output dirs."""

    name: str
    overrides: dict[str, str | int | float | bool]


def format_lr_tag(lr: float) -> str:
    """Stable directory token for learning rates (``1e-4`` → ``lr1e-4``)."""
    text = f"{lr:.0e}".replace("+", "")
    return f"lr{text}"


def default_low_lr_trials(*, num_epochs: int, batch_size: int = 64) -> list[SweepTrial]:
    """A4 grid from training_plan.md (LR only; batch fixed at 64)."""
    lrs = (1e-4, 5e-5, 2e-4)
    trials: list[SweepTrial] = []
    for lr in lrs:
        tag = format_lr_tag(lr)
        trials.append(
            SweepTrial(
                name=tag,
                overrides={
                    "train_low.num_epochs": num_epochs,
                    "train_low.lr": lr,
                    "train_low.batch_size": batch_size,
                    "train_low.output_dir": f"experiments/sweep_low/{tag}",
                    "train_low.wandb_enabled": True,
                },
            )
        )
    return trials


def default_high_lr_trials(
    *,
    num_epochs: int,
    low_checkpoint: str,
) -> list[SweepTrial]:
    """B4 default: FM learning rate only (other knobs stay at yaml defaults)."""
    trials: list[SweepTrial] = []
    for lr in (5e-5, 1e-4):
        tag = format_lr_tag(lr)
        trials.append(
            SweepTrial(
                name=tag,
                overrides={
                    "train_high.num_epochs": num_epochs,
                    "train_high.lr": lr,
                    "train_high.low_checkpoint": low_checkpoint,
                    "train_high.output_dir": f"experiments/sweep_high/{tag}",
                    "train_high.wandb_enabled": True,
                },
            )
        )
    return trials


def default_high_grid_trials(
    *,
    num_epochs: int,
    low_checkpoint: str,
) -> list[SweepTrial]:
    """Full B4 factorial (2^4 = 16 trials) — use sparingly."""
    trials: list[SweepTrial] = []
    for lr in (5e-5, 1e-4):
        for fm_hidden in (128, 256):
            for infer_steps in (5, 10):
                for embed_dim in (16, 32):
                    tag = (
                        f"{format_lr_tag(lr)}_fm{fm_hidden}"
                        f"_steps{infer_steps}_emb{embed_dim}"
                    )
                    trials.append(
                        SweepTrial(
                            name=tag,
                            overrides={
                                "train_high.num_epochs": num_epochs,
                                "train_high.lr": lr,
                                "train_high.low_checkpoint": low_checkpoint,
                                "high_policy.fm_hidden_dim": fm_hidden,
                                "high_policy.num_inference_steps": infer_steps,
                                "high_policy.option_embed_dim": embed_dim,
                                "train_high.output_dir": f"experiments/sweep_high/{tag}",
                                "train_high.wandb_enabled": True,
                            },
                        )
                    )
    return trials


def hydra_override_args(overrides: dict[str, Any]) -> list[str]:
    return [f"{key}={value}" for key, value in overrides.items()]


def to_volume_experiments_path(path: str | Path) -> str:
    """Map ``experiments/...`` to ``/experiments/...`` for Modal Volume writes."""
    text = str(path).replace("\\", "/")
    if text.startswith(f"{EXPERIMENTS_VOLUME_ROOT}/"):
        return text
    if text.startswith("experiments/"):
        return f"{EXPERIMENTS_VOLUME_ROOT}/{text[len('experiments/'):]}"
    return text


def local_mirror_for_experiments_path(path: str | Path) -> Path:
    """Local repo mirror used for post-sweep ranking (``experiments/...`` under repo root)."""
    text = str(path).replace("\\", "/")
    if text.startswith(f"{EXPERIMENTS_VOLUME_ROOT}/"):
        return REPO_ROOT / "experiments" / text[len(f"{EXPERIMENTS_VOLUME_ROOT}/") :]
    if Path(text).is_absolute():
        return Path(text)
    return REPO_ROOT / text


def modal_sweep_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Rewrite experiment paths for Modal Volume persistence."""
    out = dict(overrides)
    for key in _VOLUME_PATH_KEYS:
        if key in out:
            out[key] = to_volume_experiments_path(str(out[key]))
    return out


def trial_output_dir_key(overrides: dict[str, Any]) -> str:
    for key in _OUTPUT_DIR_KEYS:
        if key in overrides:
            return key
    raise KeyError(f"Trial overrides missing output dir key: {sorted(overrides)}")


def trial_rank_paths(trial: SweepTrial) -> tuple[Path, str]:
    """Return ``(local_mirror_dir, volume_output_dir)`` for one trial."""
    out_key = trial_output_dir_key(trial.overrides)
    local_path = str(trial.overrides[out_key])
    volume_path = to_volume_experiments_path(local_path)
    return local_mirror_for_experiments_path(local_path), volume_path


def print_sweep_ranking(ranked: list[dict[str, Any]], *, title: str = "sweep ranking (best first)") -> None:
    if not ranked:
        return
    print(f"\n=== {title} ===")
    for idx, row in enumerate(ranked, start=1):
        print(json.dumps({"rank": idx, **row}, sort_keys=True))
    print(f"\nWinner: {ranked[0]['trial']} → {ranked[0]['output_dir']}")


def print_low_sweep_rankings(ranked: dict[str, list[dict[str, Any]]]) -> None:
    """Print π_low sweep tables for total loss and diffusion loss."""
    print_sweep_ranking(
        ranked.get("val_loss", []),
        title="π_low sweep — rank by best-epoch val_loss (total)",
    )
    print_sweep_ranking(
        ranked.get("val_loss_diffusion", []),
        title="π_low sweep — rank by best-epoch val_loss_diffusion",
    )


def print_sweep_results(ranked: list[dict[str, Any]] | dict[str, list[dict[str, Any]]]) -> None:
    """Print ranking output from ``rank_low_trials`` (dual) or other sweep rankers (single list)."""
    if isinstance(ranked, dict):
        print_low_sweep_rankings(ranked)
    else:
        print_sweep_ranking(ranked)


def sync_trial_metrics_from_volume(
    *,
    volume_output_dir: str,
    local_output_dir: Path,
    volume_name: str = MODAL_VOLUME_NAME,
    dry_run: bool = False,
) -> bool:
    """Pull ``metrics.json`` from the Modal Volume into the local experiments mirror."""
    remote_metrics = f"{volume_output_dir.rstrip('/')}/metrics.json"
    local_output_dir.mkdir(parents=True, exist_ok=True)
    local_metrics = local_output_dir / "metrics.json"
    cmd = [
        "modal",
        "volume",
        "get",
        volume_name,
        remote_metrics,
        str(local_metrics),
    ]
    print(" ".join(cmd))
    if dry_run:
        return False
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError:
        print(f"Warning: could not sync {remote_metrics} from Modal Volume")
        return False
    return local_metrics.is_file()


def rank_trials_from_local_mirror(
    trials: list[SweepTrial],
    rank_fn: Callable[..., list[dict[str, Any]] | dict[str, list[dict[str, Any]]]],
    *,
    volume_dirs: dict[str, str] | None = None,
) -> list[dict[str, Any]] | dict[str, list[dict[str, Any]]]:
    """Rank trials whose ``metrics.json`` exists under the local ``experiments/`` mirror."""
    ready: list[tuple[str, Path]] = []
    missing: list[tuple[str, str]] = []
    for trial in trials:
        local_dir, volume_dir = trial_rank_paths(trial)
        if (local_dir / "metrics.json").is_file():
            ready.append((trial.name, local_dir))
        else:
            vol = (volume_dirs or {}).get(trial.name, volume_dir)
            missing.append((trial.name, vol))

    if missing:
        print("\nTrials missing local metrics.json (ranking skipped for these):")
        for name, volume_dir in missing:
            local_dir = local_mirror_for_experiments_path(volume_dir)
            print(
                f"  {name}: modal volume get {MODAL_VOLUME_NAME} "
                f"{volume_dir}/metrics.json {local_dir / 'metrics.json'}"
            )

    if not ready:
        print("\nNo local metrics available — compare trials in W&B or sync from Volume.")
        return []

    ranked = rank_fn(ready)
    print_sweep_results(ranked)
    return ranked


def run_train_script(
    script_rel: str,
    *,
    config_name: str,
    task: str = "pusht",
    overrides: dict[str, Any],
    run_readme: str,
    dry_run: bool = False,
) -> None:
    """Invoke ``python soda/training/<script>.py`` with Hydra CLI overrides."""
    from soda.experiments.run_readme import RUN_README_ENV, validate_run_readme

    config_dir = REPO_ROOT / "configs" / task
    cmd = [
        sys.executable,
        script_rel,
        "--config-path",
        str(config_dir),
        "--config-name",
        config_name,
        *hydra_override_args(overrides),
    ]
    print(" ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    env[RUN_README_ENV] = validate_run_readme(run_readme)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)


def read_metrics_history(output_dir: Path) -> list[dict[str, Any]]:
    path = Path(output_dir) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics.json: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected non-empty list in {path}")
    return data


def final_epoch_metrics(history: list[dict[str, Any]]) -> dict[str, Any]:
    return history[-1]


def best_low_epoch_metrics(
    history: list[dict[str, Any]],
    *,
    metric: str = "val_loss",
) -> dict[str, Any]:
    """Epoch with minimum ``metric`` (default total ``val_loss``)."""
    return min(history, key=lambda row: row.get(metric, float("inf")))


def summarize_low_trial(name: str, out_dir: Path) -> dict[str, Any]:
    """Per-trial summary with separate best epochs for total and diffusion val loss."""
    history = read_metrics_history(out_dir)
    best_total = best_low_epoch_metrics(history, metric="val_loss")
    best_diff = best_low_epoch_metrics(history, metric="val_loss_diffusion")
    final = final_epoch_metrics(history)
    return {
        "trial": name,
        "output_dir": str(out_dir),
        "best_total_epoch": best_total.get("epoch"),
        "best_total_val_loss": best_total.get("val_loss"),
        "best_total_val_loss_diffusion": best_total.get("val_loss_diffusion"),
        "best_total_val_loss_termination": best_total.get("val_loss_termination"),
        "best_total_val_beta_acc": best_total.get("val_beta_acc"),
        "best_diffusion_epoch": best_diff.get("epoch"),
        "best_diffusion_val_loss": best_diff.get("val_loss"),
        "best_diffusion_val_loss_diffusion": best_diff.get("val_loss_diffusion"),
        "best_diffusion_val_loss_termination": best_diff.get("val_loss_termination"),
        "best_diffusion_val_beta_acc": best_diff.get("val_beta_acc"),
        "final_epoch": final.get("epoch"),
        "final_val_loss": final.get("val_loss"),
        "final_val_loss_diffusion": final.get("val_loss_diffusion"),
    }


def _sort_low_summaries(
    summaries: list[dict[str, Any]],
    *,
    metric: Literal["val_loss", "val_loss_diffusion"],
) -> list[dict[str, Any]]:
    if metric == "val_loss":
        rows = [
            {
                "trial": s["trial"],
                "output_dir": s["output_dir"],
                "best_epoch": s["best_total_epoch"],
                "val_loss": s["best_total_val_loss"],
                "val_loss_diffusion": s["best_total_val_loss_diffusion"],
                "val_loss_termination": s["best_total_val_loss_termination"],
                "val_beta_acc": s["best_total_val_beta_acc"],
                "final_epoch": s["final_epoch"],
                "final_val_loss": s["final_val_loss"],
                "final_val_loss_diffusion": s["final_val_loss_diffusion"],
            }
            for s in summaries
        ]
        key = lambda row: (row["val_loss"] is None, row["val_loss"])
    else:
        rows = [
            {
                "trial": s["trial"],
                "output_dir": s["output_dir"],
                "best_epoch": s["best_diffusion_epoch"],
                "val_loss": s["best_diffusion_val_loss"],
                "val_loss_diffusion": s["best_diffusion_val_loss_diffusion"],
                "val_loss_termination": s["best_diffusion_val_loss_termination"],
                "val_beta_acc": s["best_diffusion_val_beta_acc"],
                "final_epoch": s["final_epoch"],
                "final_val_loss": s["final_val_loss"],
                "final_val_loss_diffusion": s["final_val_loss_diffusion"],
            }
            for s in summaries
        ]
        key = lambda row: (
            row["val_loss_diffusion"] is None,
            row["val_loss_diffusion"],
        )
    return sorted(rows, key=key)


def rank_low_trials(
    trial_dirs: list[tuple[str, Path]],
) -> dict[str, list[dict[str, Any]]]:
    """Rank π_low trials by best-epoch ``val_loss`` and ``val_loss_diffusion``."""
    summaries = [summarize_low_trial(name, out_dir) for name, out_dir in trial_dirs]
    return {
        "val_loss": _sort_low_summaries(summaries, metric="val_loss"),
        "val_loss_diffusion": _sort_low_summaries(summaries, metric="val_loss_diffusion"),
    }


def best_high_epoch_metrics(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Epoch with maximum ``val_option_acc`` (matches ``best.ckpt`` in train_high)."""
    return max(history, key=lambda row: row.get("val_option_acc", float("-inf")))


def rank_high_trials(trial_dirs: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """Rank π_high trials by best-epoch ``val_option_acc`` (higher is better)."""
    rows: list[dict[str, Any]] = []
    for name, out_dir in trial_dirs:
        history = read_metrics_history(out_dir)
        best = best_high_epoch_metrics(history)
        final = final_epoch_metrics(history)
        rows.append(
            {
                "trial": name,
                "output_dir": str(out_dir),
                "best_epoch": best.get("epoch"),
                "val_option_acc": best.get("val_option_acc"),
                "val_loss_fm": best.get("val_loss_fm"),
                "final_epoch": final.get("epoch"),
                "final_val_option_acc": final.get("val_option_acc"),
            }
        )
    rows.sort(
        key=lambda row: (
            row["val_option_acc"] is None,
            -(row["val_option_acc"] or float("-inf")),
        )
    )
    return rows


def run_sweep(
    *,
    script_rel: str,
    trials: list[SweepTrial],
    config_name: str,
    task: str = "pusht",
    rank_fn: Callable[..., list[dict[str, Any]] | dict[str, list[dict[str, Any]]]],
    run_readme: str,
    dry_run: bool = False,
    stop_on_error: bool = True,
) -> list[dict[str, Any]] | dict[str, list[dict[str, Any]]]:
    """Run each trial sequentially, then print a ranking table."""
    completed: list[tuple[str, Path]] = []
    for trial in trials:
        print(f"\n=== sweep trial: {trial.name} ===")
        try:
            run_train_script(
                script_rel,
                config_name=config_name,
                task=task,
                overrides=trial.overrides,
                run_readme=f"{run_readme} [{trial.name}]",
                dry_run=dry_run,
            )
        except subprocess.CalledProcessError:
            if stop_on_error:
                raise
            print(f"Trial {trial.name} failed — skipping ranking for this trial.")
            continue
        local_dir, _ = trial_rank_paths(trial)
        completed.append((trial.name, local_dir))

    if dry_run or not completed:
        print("\n(dry-run or no completed trials — ranking skipped)")
        return []

    ranked = rank_fn(completed)
    print_sweep_results(ranked)
    return ranked


def run_modal_sweep(
    *,
    modal_script: Literal["modal_train_low.py", "modal_train_high.py"],
    trials: list[SweepTrial],
    config_name: str,
    task: str,
    rank_fn: Callable[..., list[dict[str, Any]] | dict[str, list[dict[str, Any]]]],
    run_readme: str,
    detach: bool = True,
    dry_run: bool = False,
    stop_on_error: bool = True,
) -> list[dict[str, Any]] | dict[str, list[dict[str, Any]]]:
    """Launch sweep trials on Modal, optionally sync metrics and rank locally."""
    volume_dirs: dict[str, str] = {}
    for trial in trials:
        print(f"\n=== Modal sweep trial: {trial.name} ===")
        modal_overrides = modal_sweep_overrides(trial.overrides)
        local_dir, volume_dir = trial_rank_paths(trial)
        volume_dirs[trial.name] = volume_dir
        print(f"Volume output: {volume_dir}")
        print(f"Local mirror:  {local_dir}")
        try:
            spawn_modal_train(
                modal_script=modal_script,
                config_name=config_name,
                task=task,
                hydra_overrides=modal_overrides,
                run_readme=f"{run_readme} [{trial.name}]",
                detach=detach,
                dry_run=dry_run,
            )
        except subprocess.CalledProcessError:
            if stop_on_error:
                raise
            print(f"Trial {trial.name} failed — continuing.")
            continue

        if dry_run or detach:
            continue

        sync_trial_metrics_from_volume(
            volume_output_dir=volume_dir,
            local_output_dir=local_dir,
            dry_run=dry_run,
        )

    if dry_run:
        print("\n(dry-run — Modal jobs not launched; ranking skipped)")
        return []

    if detach:
        print(
            "\nModal jobs launched detached. Artifacts persist on Volume under "
            f"{EXPERIMENTS_VOLUME_ROOT}/sweep_low/ or {EXPERIMENTS_VOLUME_ROOT}/sweep_high/."
        )
        print(
            "After jobs finish, sync metrics and rank locally, e.g.:\n"
            f"  modal volume get {MODAL_VOLUME_NAME} "
            f"{EXPERIMENTS_VOLUME_ROOT}/sweep_low/<trial>/metrics.json "
            f"experiments/sweep_low/<trial>/metrics.json\n"
            "Then re-run ranking with W&B or pull all trial dirs from the Volume."
        )
        return []

    return rank_trials_from_local_mirror(trials, rank_fn, volume_dirs=volume_dirs)


def spawn_modal_train(
    *,
    modal_script: Literal["modal_train_low.py", "modal_train_high.py"],
    config_name: str,
    task: str,
    hydra_overrides: dict[str, Any],
    run_readme: str,
    detach: bool = True,
    dry_run: bool = False,
) -> None:
    """Run one trial on Modal (uses ``modal run modal/<script> --hydra-overrides ...``).

    Callers should pass Volume paths (``/experiments/...``); use ``modal_sweep_overrides``
    when starting from local ``experiments/...`` dirs.
    """
    override_text = " ".join(hydra_override_args(hydra_overrides))
    cmd = [
        "modal",
        "run",
        f"modal/{modal_script}",
        "--config-name",
        config_name,
        "--task",
        task,
        "--hydra-overrides",
        override_text,
        "--run-readme",
        run_readme,
    ]
    if detach:
        cmd.append("--detach")
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
