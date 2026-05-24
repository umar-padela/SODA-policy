#!/usr/bin/env python3
"""
π_high hyperparameter sweep (training_plan.md B4).

Default: FM LR only (``5e-5``, ``1e-4``). Use ``--full-grid`` for the 16-trial
factorial over LR × ``fm_hidden_dim`` × ``num_inference_steps`` × ``option_embed_dim``.

Ranking uses **best-epoch** ``val_option_acc`` (same as ``best.ckpt``).
Default **14 epochs** per trial (post-warmup cosine steps after ``lr_warmup_epochs=5``).

Examples:

  python scripts/sweep_train_high.py \\
    --low-checkpoint experiments/train_low/soda_supervised/best.ckpt \\
    --num-epochs 14 --dry-run --run-readme "B4 pi_high LR sweep"

  python scripts/sweep_train_high.py \\
    --low-checkpoint experiments/train_low/soda_supervised/best.ckpt \\
    --num-epochs 14 --modal --no-detach --run-readme "B4 on Modal"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from soda.training.sweep_common import (  # noqa: E402
    default_high_grid_trials,
    default_high_lr_trials,
    rank_high_trials,
    run_modal_sweep,
    run_sweep,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="soda_supervised")
    parser.add_argument("--task", default="pusht")
    parser.add_argument(
        "--low-checkpoint",
        required=True,
        help="Fixed π_low best.ckpt (local ``experiments/...`` or Modal ``/experiments/...``)",
    )
    parser.add_argument("--num-epochs", type=int, default=14)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run full B4 factorial (16 trials)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--modal", action="store_true")
    parser.add_argument(
        "--no-detach",
        action="store_true",
        help="With --modal, wait for each job and sync metrics for local ranking",
    )
    parser.add_argument(
        "--run-readme",
        required=True,
        help="Required description stored in each trial's archived run README",
    )
    args = parser.parse_args()

    if args.full_grid:
        trials = default_high_grid_trials(
            num_epochs=args.num_epochs,
            low_checkpoint=args.low_checkpoint,
        )
    else:
        trials = default_high_lr_trials(
            num_epochs=args.num_epochs,
            low_checkpoint=args.low_checkpoint,
        )

    if args.modal:
        run_modal_sweep(
            modal_script="modal_train_high.py",
            trials=trials,
            config_name=args.config_name,
            task=args.task,
            rank_fn=rank_high_trials,
            run_readme=args.run_readme,
            detach=not args.no_detach,
            dry_run=args.dry_run,
        )
        return 0

    run_sweep(
        script_rel="soda/training/train_high.py",
        trials=trials,
        config_name=args.config_name,
        task=args.task,
        rank_fn=rank_high_trials,
        run_readme=args.run_readme,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
