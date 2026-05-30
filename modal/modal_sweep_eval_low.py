"""
Local entrypoint — sweep eval over π_low checkpoints from a completed training run.

Discovers every ``epoch_NNNN.ckpt`` file in a Modal Volume directory, keeps those
whose epoch number is a multiple of ``epoch_stride`` or in ``pinned_epochs``, spawns
one parallel GPU job per checkpoint (reusing ``rollout_hierarchical``), collects
scores, prints a table, and optionally saves a JSON summary.

Examples (repo root)::

  # Standard sweep: every 50th epoch + pinned 150/175, 50 episodes each
  modal run modal/modal_sweep_eval_low.py \\
    --low-dir pusht/train_low/soda_supervised \\
    --high-checkpoint pusht/train_high/soda_supervised/best.ckpt

  # Quick smoke: every 50th epoch, 10 episodes
  modal run modal/modal_sweep_eval_low.py \\
    --low-dir pusht/train_low/soda_supervised \\
    --high-checkpoint pusht/train_high/soda_supervised/best.ckpt \\
    --n-episodes 10

  # Save summary JSON locally
  modal run modal/modal_sweep_eval_low.py \\
    --low-dir pusht/train_low/soda_supervised \\
    --high-checkpoint pusht/train_high/soda_supervised/best.ckpt \\
    --output tmp/sweep_summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import modal

from modal_config import app, rollout_hierarchical
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path


def _discover_checkpoints(
    vol: modal.Volume,
    low_dir: str,
    epoch_stride: int,
    min_epoch: int,
    pinned_epochs: list[int],
) -> list[tuple[int, str]]:
    """Return sorted (epoch_num, abs_volume_path) for matching epoch_NNNN.ckpt files.

    Includes epochs where ``epoch_num >= min_epoch AND epoch_num % epoch_stride == 0``,
    plus any epochs in ``pinned_epochs`` (always included regardless of stride/min).
    """
    rel_dir = volume_relative_path(low_dir)
    entries = vol.listdir(rel_dir)

    pinned_set = set(pinned_epochs)
    epoch_ckpts: list[tuple[int, str]] = []
    for entry in entries:
        fname = Path(entry.path).name
        m = re.match(r"epoch_(\d{4})\.ckpt$", fname)
        if not m:
            continue
        epoch_num = int(m.group(1))
        stride_hit = epoch_num >= min_epoch and epoch_num % epoch_stride == 0
        if stride_hit or epoch_num in pinned_set:
            abs_path = f"{low_dir.rstrip('/')}/{fname}"
            epoch_ckpts.append((epoch_num, abs_path))

    epoch_ckpts.sort()
    return epoch_ckpts


@app.local_entrypoint()
def main(
    low_dir: str,
    high_checkpoint: str,
    config: str = "configs/pusht/soda_supervised.yaml",
    epoch_stride: int = 50,
    min_epoch: int = 200,
    pinned_epochs: str = "150,175",
    n_episodes: int = 50,
    beta_transition: float | None = None,
    output: str | None = None,
):
    """
    Sweep eval over π_low checkpoints on Modal.

    Parameters
    ----------
    low_dir
        Volume path to the low-policy training directory, e.g.
        ``pusht/train_low/soda_supervised`` or
        ``/experiments/pusht/train_low/soda_supervised``.
    high_checkpoint
        Volume path to the π_high checkpoint, e.g.
        ``pusht/train_high/train_high_multianchor/best.ckpt``.
    config
        Path to SODA yaml for env + inference settings.
    epoch_stride
        Evaluate every N epochs among epochs >= min_epoch (default 50).
    min_epoch
        First epoch included by stride selection (default 200).
        Pinned epochs are always included regardless of this floor.
    pinned_epochs
        Comma-separated epochs always included regardless of stride/min_epoch
        (default ``150,175`` → two-phase boundary checkpoints).
    n_episodes
        Episodes per checkpoint (default 50 = full eval; use 10 for a quick smoke).
    beta_transition
        Override beta_transition from yaml; None uses yaml value.
    output
        Local path to save JSON summary (optional).
    """
    pinned_list = [int(e.strip()) for e in pinned_epochs.split(",") if e.strip()]

    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    epoch_ckpts = _discover_checkpoints(vol, low_dir, epoch_stride, min_epoch, pinned_list)

    if not epoch_ckpts:
        print(
            f"No epoch_NNNN.ckpt checkpoints found in {low_dir} "
            f"(stride={epoch_stride}, pinned={pinned_list})."
        )
        return

    print(
        f"Found {len(epoch_ckpts)} checkpoints to evaluate "
        f"(epochs: {[e for e, _ in epoch_ckpts]})"
    )

    # Spawn all jobs in parallel — each gets its own Modal GPU instance.
    calls = []
    for epoch, low_ckpt_path in epoch_ckpts:
        call = rollout_hierarchical.spawn(
            config_path=config,
            high_checkpoint=high_checkpoint,
            low_checkpoint=low_ckpt_path,
            n_episodes=n_episodes,
            test_start_seed=100000,
            n_action_steps=8,
            beta_transition=beta_transition,
            no_video=True,
        )
        calls.append((epoch, low_ckpt_path, call))

    print(f"Spawned {len(calls)} rollout jobs — waiting for results ...")

    rows = []
    for epoch, low_ckpt_path, call in calls:
        result = call.get()
        mean_score = float(result.get("mean_max_overlap", 0.0))
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        std_score = float(np.std(scores)) if scores else 0.0
        rows.append(
            {
                "epoch": epoch,
                "mean_max_overlap": mean_score,
                "std": std_score,
                "low_checkpoint": low_ckpt_path,
                "n_episodes": len(episodes),
            }
        )
        print(f"  epoch {epoch:4d}: mean={mean_score:.1f}%  std={std_score:.1f}%")

    print("\n=== Sweep eval summary ===")
    print(f"{'Epoch':>6}  {'Mean%':>7}  {'Std%':>6}  Checkpoint")
    for row in sorted(rows, key=lambda r: r["epoch"]):
        fname = Path(row["low_checkpoint"]).name
        print(f"{row['epoch']:>6}  {row['mean_max_overlap']:>7.1f}  {row['std']:>6.1f}  {fname}")

    best = max(rows, key=lambda r: r["mean_max_overlap"])
    print(
        f"\nBest: epoch {best['epoch']} → {best['mean_max_overlap']:.1f}%"
        f"  ({Path(best['low_checkpoint']).name})"
    )

    if output:
        summary = {
            "high_checkpoint": high_checkpoint,
            "low_dir": low_dir,
            "epoch_stride": epoch_stride,
            "min_epoch": min_epoch,
            "pinned_epochs": pinned_list,
            "n_episodes": n_episodes,
            "rows": rows,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(summary, indent=2))
        print(f"Saved summary → {output}")
