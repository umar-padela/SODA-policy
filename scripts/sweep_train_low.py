#!/usr/bin/env python3
"""
π_low LR sweep (training_plan.md A4).

Runs seven log-spaced trials from ``3e-5`` to ``5e-4`` (batch size 64 fixed),
then prints **two rankings** (best-epoch ``val_loss`` and ``val_loss_diffusion``).

W&B: project ``soda-sweep-low``, run name ``lr3.00e-05`` (etc.), group ``A4-lr-sweep``.

Default **14 epochs** so trials see post-warmup cosine steps (``lr_warmup_epochs=5``).

Examples (repo root):

  python scripts/sweep_train_low.py --num-epochs 14 --dry-run --run-readme "A4 pi_low LR sweep"
  python scripts/sweep_train_low.py --num-epochs 14 --run-readme "A4 pi_low LR sweep"
  python scripts/sweep_train_low.py --num-epochs 14 --modal --no-detach --run-readme "A4 on Modal"
  python scripts/sweep_train_low.py --num-epochs 14 --modal --detach --run-readme "A4 on Modal"
  python scripts/sweep_train_low.py --rank-only --sync-modal --num-epochs 14
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
    rank_sweep_trials,
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
        "--rank-only",
        action="store_true",
        help="Skip training; rank from local metrics.json (use with --sync-modal after Modal)",
    )
    parser.add_argument(
        "--sync-modal",
        action="store_true",
        help="Before ranking, pull metrics.json from Modal Volume into experiments/",
    )
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
        "--detach",
        action="store_true",
        help="Spawn one sequential Modal job; laptop can disconnect (rank later with --rank-only)",
    )
    parser.add_argument(
        "--no-detach",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-readme",
        default=None,
        help="Description stored in each trial's archived run README (required unless --rank-only)",
    )
    args = parser.parse_args()

    if not args.rank_only and not args.run_readme:
        parser.error("--run-readme is required unless --rank-only")

    trials = default_low_lr_trials(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        task=args.task,
    )

    if args.rank_only:
        rank_sweep_trials(
            trials,
            rank_low_trials,
            sync_modal=args.sync_modal,
            dry_run=args.dry_run,
        )
        return 0

    if args.detach and args.no_detach:
        parser.error("Use only one of --detach and --no-detach")

    if args.modal:
        run_modal_sweep(
            modal_script="modal_train_low.py",
            trials=trials,
            config_name=args.config_name,
            task=args.task,
            rank_fn=rank_low_trials,
            run_readme=args.run_readme,
            detach=args.detach and not args.no_detach,
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
