"""
Train π_high with flow matching on option segment starts (project_plan §7 row 15).

Pairs with ``soda/models/high_policy.py`` and ``soda/training/losses_high.py``.

Training loop pattern: ``assignment1/hw1/main.py::train_flow_matching_policy``
(generic ``train_policy(policy, flow_matching_option_loss, ...)``).

Entrypoints
-----------
- Local: ``python soda/training/train_high.py --config-path configs/pusht --config-name soda_supervised``
- Modal: ``modal run modal/modal_train_high.py``

Hydra
-----
- [ ] Add ``high_policy:`` and ``train_high:`` to ``configs/pusht/soda_supervised.yaml``
- [ ] Reuse ``task.dataset.zarr_path``, ``option_id_key`` from same zarr as π_low
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

__all__ = [
    "HighPolicy",
    "HighPolicyConfig",
    "OptionStartDataset",
    "flow_matching_option_loss",
    "main",
]


@dataclass
class TrainHighConfig:
    """
    Training loop settings (mirror ``assignment1/hw1/main.py`` train kwargs; wire from YAML).

    TODO — ``train_high:`` block in config
    --------------------------------------
    - [ ] ``device``, ``seed``, ``num_epochs``, ``batch_size``, ``lr``, ``weight_decay``
    - [ ] ``checkpoint_every``, ``output_dir`` (Modal: ``/experiments/train_high/``)
    - [ ] ``wandb_project``, ``wandb_enabled`` (optional)
    - [ ] ``val_ratio`` — same episode split as ``OptionLabeledZarrDataset``
    """

    num_epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-4
    device: str = "cuda:0"


class OptionStartDataset:
    """
    One sample per option segment: obs window at **segment.start**, label ω.

    Index unit = ``OptionSegment`` from ``build_option_segment_index`` (not HW1 sliding
    windows over all frames). See ``option_aware_dataset._obs_window`` for obs layout.

    TODO §7 row 15
    --------------
    - [ ] Zarr + ``option_id_key`` + episode train/val mask (reuse dataset helpers)
    - [ ] ``anchor = segment.start`` always (never random interior — π_high only)
    - [ ] ``__getitem__`` → ``{"obs": {image, agent_pos}, "option_id": long}``
    - [ ] ``get_normalizer()`` for π_low / eval consistency (optional at M1)
    """

    def __init__(self, zarr_path: str | Path, **kwargs: Any) -> None:
        raise NotImplementedError("TODO §7 row 15: OptionStartDataset")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict[str, Any]:
        raise NotImplementedError


def _batch_option_loss(
    policy: HighPolicy,
    batch: dict[str, Any],
) -> Any:
    """
    Single-batch FM loss (HW1: one call to ``flow_matching_loss`` inside train loop).

    TODO §7 row 15
    --------------
    - [ ] ``global_feat = policy.encode_obs(batch["obs"])``
    - [ ] ``return flow_matching_option_loss(policy, global_feat, batch["option_id"])``
    """
    raise NotImplementedError("TODO §7 row 15: _batch_option_loss")


def train_one_epoch(
    policy: HighPolicy,
    loader: Any,
    optimizer: Any,
    device: str,
) -> dict[str, float]:
    """
    One epoch over segment-start batches.

    TODO §7 row 15 (mirror ``assignment1/hw1`` generic train loop)
    -----------------------------------------------------------
    - [ ] ``policy.train()``
    - [ ] for batch in loader: move to device → ``loss = _batch_option_loss(...)``
    - [ ] ``loss.backward()``; optimizer.step(); zero_grad
    - [ ] return ``{"loss_fm": mean}``  (no diffusion / termination terms)
    """
    raise NotImplementedError("TODO §7 row 15: train_one_epoch")


def validate(
    policy: HighPolicy,
    loader: Any,
    device: str,
) -> dict[str, float]:
    """
    M1 validation on held-out segment starts (§7 row 17).

    TODO §7 row 15
    --------------
    - [ ] ``loss_fm``: same ``_batch_option_loss`` under ``torch.no_grad()``
    - [ ] ``option_acc``: ``sample_option(encode_obs(obs))`` vs ``option_id``
    - [ ] optional confusion matrix / per-class recall (log outside this file)
    """
    raise NotImplementedError("TODO §7 row 15: validate")


def build_policy_and_optimizer(cfg: Any) -> tuple[HighPolicy, Any]:
    """
    Build ``HighPolicy`` + optimizer from Hydra config.

    TODO §7 row 15
    --------------
    - [ ] ``num_options`` from config or ``np.unique(zarr[option_id_key])``
    - [ ] ``HighPolicyConfig(**cfg.high_policy)``
    - [ ] AdamW on all π_high parameters
    """
    raise NotImplementedError("TODO §7 row 15: build_policy_and_optimizer")


def save_checkpoint(
    path: Path,
    policy: HighPolicy,
    optimizer: Any,
    epoch: int,
    cfg: Any,
) -> None:
    """
    Save checkpoint for later ``HierarchicalController`` / eval loaders.

    TODO §7 row 15
    --------------
    - [ ] torch.save dict: policy state, optimizer, epoch, cfg snapshot
    """
    raise NotImplementedError("TODO §7 row 15: save_checkpoint")


def run_training(cfg: Any) -> None:
    """
    Full train loop (Hydra ``main`` calls this).

    TODO §7 row 15
    --------------
    - [ ] Seed, device, ``OptionStartDataset`` train/val DataLoaders
    - [ ] Epoch loop: ``train_one_epoch`` → ``validate`` → save best by ``option_acc``
    """
    raise NotImplementedError("TODO §7 row 15: run_training")


def main() -> None:
    """
    Hydra CLI entrypoint.

    TODO §7 row 15
    --------------
    - [ ] ``@hydra.main(config_path="../../configs/pusht", config_name="soda_supervised")``
    - [ ] ``run_training(cfg)``
    """
    raise NotImplementedError(
        "TODO §7 row 15: Hydra main — "
        "`python soda/training/train_high.py --config-path configs/pusht "
        "--config-name soda_supervised`"
    )


if __name__ == "__main__":
    main()
