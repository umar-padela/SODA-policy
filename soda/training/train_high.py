"""
Train π_high with flow matching on option segment starts (project_plan §7 row 15).

Pairs with ``soda/models/high_policy.py`` and ``soda/training/losses_high.py``.

Entrypoints
-----------
- Local: ``python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised``
- Modal: ``modal run modal/modal_train_high.py``

Hydra
-----
- [ ] Add / fill ``configs/pusht/soda_supervised.yaml`` section ``train_high:`` and
      ``high_policy:`` (see ``HighPolicyConfig`` fields)
- [ ] Reuse ``dataset.zarr_path``, ``dataset.option_id_key`` from same zarr as π_low
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# TODO §7 row 15:
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch
# from torch.utils.data import DataLoader

from soda.models.high_policy import HighPolicy, HighPolicyConfig
from soda.training.losses_high import flow_matching_option_loss

__all__ = ["flow_matching_option_loss", "main"]


@dataclass
class TrainHighConfig:
    """
    Training loop settings (mirror structure you will add to YAML).

    TODO — wire from Hydra
    ----------------------
    - [ ] ``device``, ``seed``, ``num_epochs``, ``batch_size``, ``lr``, ``weight_decay``
    - [ ] ``checkpoint_every``, ``output_dir`` (Modal: under ``/experiments/train_high/``)
    - [ ] ``wandb_project``, ``wandb_enabled`` (optional; match train_low pattern)
    - [ ] ``val_ratio`` — episode split consistent with ``OptionLabeledZarrDataset``
    """

    num_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cuda:0"


class OptionStartDataset:
    """
    One sample per option segment: observation window at **segment start**, label ω.

    Differs from ``OptionLabeledZarrDataset`` training samples (random anchor + full
    stretched actions). π_high only needs ``(obs, option_id)`` at segment boundaries.

    TODO §7 row 15
    --------------
    - [ ] Implement by wrapping zarr + ``build_option_segment_index`` (preferred), or
    - [ ] Subclass / compose ``OptionLabeledZarrDataset`` with ``random_anchor=False``
          and override ``__getitem__`` to fix anchor = segment.start
    - [ ] ``__getitem__`` returns ``{"obs": {...}, "option_id": int}`` (torch tensors)
    - [ ] ``__len__`` = number of segments (train episode mask)
    - [ ] Share ``get_normalizer`` / image stats with π_low for consistency
    """

    def __init__(self, zarr_path: str | Path, **kwargs: Any) -> None:
        raise NotImplementedError("TODO §7 row 15: OptionStartDataset")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError


def train_one_epoch(
    policy: HighPolicy,
    loader: Any,
    optimizer: Any,
    device: str,
) -> dict[str, float]:
    """
    Single training epoch over segment-start batches.

    TODO §7 row 15
    --------------
    - [ ] Loop batches; ``encode_obs`` → ``flow_matching_option_loss``
    - [ ] ``loss.backward()``; grad clip optional
    - [ ] Log ``loss_fm`` mean; no termination / diffusion terms here
    """
    raise NotImplementedError("TODO §7 row 15: train_one_epoch")


def validate(
    policy: HighPolicy,
    loader: Any,
    device: str,
) -> dict[str, float]:
    """
    Validation metrics for π_high.

    TODO §7 row 15
    --------------
    - [ ] Loss: same FM loss on val segment starts
    - [ ] Accuracy: ``sample_option`` vs ground-truth ω (greedy, no grad)
    - [ ] Log ``val/loss_fm``, ``val/option_acc``
    """
    raise NotImplementedError("TODO §7 row 15: validate")


def build_policy_and_optimizer(cfg: Any) -> tuple[HighPolicy, Any]:
    """
    Construct ``HighPolicy`` + AdamW from Hydra config.

    TODO §7 row 15
    --------------
    - [ ] Resolve ``num_options`` from zarr metadata or config
    - [ ] ``HighPolicyConfig`` from ``cfg.high_policy``
    - [ ] Optimizer on all π_high parameters
    """
    raise NotImplementedError("TODO §7 row 15: build_policy_and_optimizer")


def save_checkpoint(
    path: Path,
    policy: HighPolicy,
    optimizer: Any,
    epoch: int,
    cfg: Any,
) -> None:
    """Save ``.ckpt`` compatible with future ``load_high_policy`` in eval/inference."""
    raise NotImplementedError("TODO §7 row 15: save_checkpoint")


def run_training(cfg: Any) -> None:
    """
    Full train loop (called from Hydra ``main``).

    TODO §7 row 15
    --------------
    - [ ] Set seed, device
    - [ ] ``OptionStartDataset`` train + val loaders
    - [ ] Epoch loop: train_one_epoch → validate → save best
    - [ ] Final checkpoint to ``cfg.output_dir`` / Modal volume
    - [ ] Optional W&B init (same secret as ``modal_config`` ``wandb``)
    """
    raise NotImplementedError("TODO §7 row 15: run_training")


def main() -> None:
    """
    Hydra CLI entrypoint.

    TODO §7 row 15
    --------------
    - [ ] ``@hydra.main(config_path="../../configs/pusht", config_name="soda_supervised")``
    - [ ] ``run_training(cfg)``
    - [ ] Document overrides: ``train_high.num_epochs=...``, ``high_policy.num_options=...``
    """
    raise NotImplementedError(
        "TODO §7 row 15: Hydra main — run via "
        "`python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised`"
    )


if __name__ == "__main__":
    main()
