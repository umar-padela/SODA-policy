#!/usr/bin/env python3
"""
π_high LR sweep (training_plan.md B4).

Five FM learning rates: ``1e-5``, ``5e-5``, ``1e-4``, ``5e-4``, ``1e-3``.
Architecture stays at yaml defaults in ``configs/pusht/soda_supervised.yaml``:
``fm_hidden_dim=256``, ``num_inference_steps=10``, ``option_embed_dim=32``.

W&B: project ``soda-sweep-high``, run name ``lr1.00e-04`` (etc.), group ``B4-lr-sweep``.

Ranking uses **best-epoch** ``val_option_acc`` (same as ``best.ckpt``).
Default **14 epochs** per trial (post-warmup cosine steps after ``lr_warmup_epochs=5``).

Examples:

  python scripts/sweep_train_high.py \\
    --low-checkpoint experiments/pusht/train_low/soda_supervised/best.ckpt \\
    --num-epochs 14 --dry-run --run-readme "B4 pi_high LR sweep"

  python scripts/sweep_train_high.py \\
    --low-checkpoint /experiments/pusht/train_low/soda_supervised/best.ckpt \\
    --num-epochs 14 --modal --no-detach --run-readme "B4 pi_high LR sweep on Modal"

  python scripts/sweep_train_high.py \\
    --low-checkpoint /experiments/pusht/train_low/soda_supervised/best.ckpt \\
    --rank-only --sync-modal --num-epochs 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from soda.training.sweep_common import (  # noqa: E402
    default_high_lr_trials,
    rank_high_trials,
    rank_sweep_trials,
    run_modal_sweep,
    run_sweep,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="soda_supervised")
    parser.add_argument("--task", default="pusht")
    parser.add_argument(
        "--low-checkpoint",
        default=None,
        help="Fixed π_low best.ckpt (required unless --rank-only)",
    )
    parser.add_argument("--num-epochs", type=int, default=14)
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--modal", action="store_true")
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

    if not args.rank_only and not args.low_checkpoint:
        parser.error("--low-checkpoint is required unless --rank-only")
    if not args.rank_only and not args.run_readme:
        parser.error("--run-readme is required unless --rank-only")

    low_ckpt = args.low_checkpoint or "experiments/pusht/train_low/soda_supervised/best.ckpt"
    trials = default_high_lr_trials(
        num_epochs=args.num_epochs,
        low_checkpoint=low_ckpt,
        task=args.task,
    )

    if args.rank_only:
        rank_sweep_trials(
            trials,
            rank_high_trials,
            sync_modal=args.sync_modal,
            dry_run=args.dry_run,
        )
        return 0

    if args.detach and args.no_detach:
        parser.error("Use only one of --detach and --no-detach")

    if args.modal:
        run_modal_sweep(
            modal_script="modal_train_high.py",
            trials=trials,
            config_name=args.config_name,
            task=args.task,
            rank_fn=rank_high_trials,
            run_readme=args.run_readme,
            detach=args.detach and not args.no_detach,
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
