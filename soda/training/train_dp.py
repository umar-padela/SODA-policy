"""
Train vanilla Diffusion Policy on Push-T from ``configs/pusht/dp.yaml``.

Uses Columbia ``third_party/diffusion_policy`` workspace (same stack as frozen eval).
Checkpoints land in ``train_dp.output_dir`` (default ``experiments/train_dp/dp/``).

Entrypoints
-----------
- Local: ``python soda/training/train_dp.py --config-path configs/pusht --config-name dp``
- Modal: ``modal run modal/modal_train_dp.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soda.training.dp_config import (
    _cfg_get,
    _get_block,
    build_columbia_workspace_cfg,
    resolve_dp_output_dir,
)

__all__ = ["run_training", "main"]


def _require_diffusion_policy() -> None:
    from soda.eval.policy_loaders import _require_diffusion_policy as _req

    _req()


def run_training(yaml_cfg: Any) -> Path:
    """Build Columbia workspace from ``dp.yaml`` and run training."""
    import hydra

    from soda.experiments.run_readme import (
        begin_training_run_archive,
        finalize_training_run_archive,
    )

    _require_diffusion_policy()
    workspace_cfg = build_columbia_workspace_cfg(yaml_cfg)
    out_dir = resolve_dp_output_dir(yaml_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _get_block(yaml_cfg, "train_dp")
    run_archive = begin_training_run_archive(
        out_dir,
        kind="train_dp",
        hydra_readme=_cfg_get(train, "run_readme", None),
        extra_metadata={
            "config_name": str(_cfg_get(yaml_cfg, "name", "dp")),
            "num_epochs": int(_cfg_get(train, "num_epochs", 3050)),
        },
    )

    cls = hydra.utils.get_class(workspace_cfg._target_)
    workspace = cls(workspace_cfg, output_dir=str(out_dir))
    workspace.run()
    finalize_training_run_archive(run_archive)
    return out_dir


def main(cfg: Any) -> None:
    run_training(cfg)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(
        version_base=None,
        config_path="../../configs/pusht",
        config_name="dp",
    )
    def _cli(cfg: DictConfig) -> None:
        main(cfg)

    _cli()
