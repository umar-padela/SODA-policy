#!/usr/bin/env python3
"""
π_low LR sweep (training_plan.md A4).

Runs three short trials (``1e-4``, ``5e-5``, ``2e-4``) with batch size 64 fixed,
then prints **two rankings** (best-epoch ``val_loss`` and ``val_loss_diffusion``).

Default **14 epochs** so trials see post-warmup cosine steps (``lr_warmup_epochs=5``).

Examples (repo root):

  python scripts/sweep_train_low.py --num-epochs 14 --dry-run --run-readme "A4 pi_low LR sweep"
  python scripts/sweep_train_low.py --num-epochs 14 --run-readme "A4 pi_low LR sweep"
  python scripts/sweep_train_low.py --num-epochs 14 --modal --no-detach --run-readme "A4 on Modal"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from soda.training.sweep_common import (  # noqa: E402
    default_low_lr_trials,
    rank_low_trials,
    run_modal_sweep,
    run_sweep,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="soda_supervised")
    parser.add_argument("--task", default="pusht")
    parser.add_argument("--num-epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only (no training)",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Run each trial via ``modal run modal/modal_train_low.py``",
    )
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

    trials = default_low_lr_trials(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    if args.modal:
        run_modal_sweep(
            modal_script="modal_train_low.py",
            trials=trials,
            config_name=args.config_name,
            task=args.task,
            rank_fn=rank_low_trials,
            run_readme=args.run_readme,
            detach=not args.no_detach,
            dry_run=args.dry_run,
        )
        return 0

    run_sweep(
        script_rel="soda/training/train_low.py",
        trials=trials,
        config_name=args.config_name,
        task=args.task,
        rank_fn=rank_low_trials,
        run_readme=args.run_readme,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
