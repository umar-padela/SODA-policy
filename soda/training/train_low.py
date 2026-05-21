"""
Train π_low (diffusion + termination) — project_plan §7 row 14.

Uses ``soda/training/losses_low.py`` for ``L_diffusion + L_termination``.
Pairs with ``soda/models/low_policy.py`` and ``OptionLabeledZarrDataset``.

Modal: ``modal run modal/modal_train_low.py``

TODO §7 row 14 — mirror structure in ``train_high.py`` (Hydra, checkpoint, W&B).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("TODO §7 row 14: train_low Hydra workspace")


if __name__ == "__main__":
    main()
