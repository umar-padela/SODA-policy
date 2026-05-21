"""
Generic Push-T eval orchestrator (project_plan §6, §7 rows 8–9, 17, 22).

Loads a policy, runs sim rollouts, computes overlap metrics at multiple step horizons.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from soda.eval.metrics import DEFAULT_OVERLAP_CHECKPOINTS
from soda.eval.run_naming import (
    build_eval_output_dir,
    build_eval_run_dir_name,
    checkpoint_slug,
    policy_label_from_config,
)
from soda.eval.policy_loaders import (
    load_dp_image_policy,
    load_soda_policy,
)
from soda.eval.pusht_rollout import run_pusht_rollouts
from soda.inference.control_regimes import ControlRegime

PolicySource = Literal["dp_baseline", "soda"]
TaskName = Literal["pusht", "square"]


@dataclass
class EvalConfig:
    """Configuration for a Push-T eval run."""

    checkpoint_path: Path
    task: TaskName = "pusht"
    policy_source: PolicySource = "dp_baseline"
    ckpt_slug: str | None = None
    device: str = "cuda:0"
    regime: ControlRegime = "receding"
    action_horizon: int = 1
    max_steps: int = 300
    overlap_checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS
    n_test: int = 50
    n_train: int = 0
    test_start_seed: int = 10000
    train_start_seed: int = 0
    # SODA-only (when implemented)
    soda_config_name: str = "soda_supervised"
    show_progress: bool = True


def _resolve_checkpoint(config: EvalConfig) -> Path:
    """Require an explicit checkpoint path (DP frozen, your DP, or SODA)."""
    path = Path(config.checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Pass --checkpoint to eval. For Columbia frozen DP on Modal, run "
            "`modal run modal/modal_download_dp.py` once, then use the printed path."
        )
    return path


def load_policy_for_eval(
    config: EvalConfig,
    checkpoint: Path,
) -> tuple[Any, Any]:
    """Return ``(policy, settings)`` for rollout."""
    if config.policy_source == "dp_baseline":
        return load_dp_image_policy(checkpoint, device=config.device)
    if config.policy_source == "soda":
        return load_soda_policy(
            checkpoint,
            device=config.device,
            config_name=config.soda_config_name,
        )
    raise ValueError(f"Unknown policy_source: {config.policy_source}")


def resolve_eval_output_dir(
    config: EvalConfig,
    experiments_root: Path,
    *,
    ckpt_slug: str | None = None,
) -> Path:
    """Build ``{root}/eval/{task}/{descriptive_run_name}/``."""
    return build_eval_output_dir(
        experiments_root,
        task=config.task,
        policy_source=config.policy_source,
        soda_config_name=config.soda_config_name,
        action_horizon=config.action_horizon,
        regime=config.regime,
        n_test=config.n_test,
        max_steps=config.max_steps,
        checkpoint=Path(config.checkpoint_path),
        ckpt_slug=ckpt_slug or config.ckpt_slug,
        test_start_seed=config.test_start_seed,
    )


def run_pusht_eval(config: EvalConfig, output_dir: Path) -> dict[str, Any]:
    """
    End-to-end eval: load checkpoint → rollouts → write JSON.

    ``policy_source`` selects the loader (``dp_baseline`` vs ``soda``); the
    checkpoint path is always explicit so frozen DP, retrained DP, and SODA
    share the same interface.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _resolve_checkpoint(config)
    policy, settings = load_policy_for_eval(config, checkpoint)

    # Honor eval horizon from config (project uses 300 steps).
    settings_max = config.max_steps

    summary = run_pusht_rollouts(
        policy,
        settings,
        n_test=config.n_test,
        n_train=config.n_train,
        test_start_seed=config.test_start_seed,
        train_start_seed=config.train_start_seed,
        max_steps=settings_max,
        regime=config.regime,
        action_horizon=config.action_horizon,
        checkpoints=config.overlap_checkpoints,
        show_progress=config.show_progress,
    )

    policy_label = policy_label_from_config(
        config.policy_source, config.soda_config_name
    )
    run_dir_name = build_eval_run_dir_name(
        task=config.task,
        policy_label=policy_label,
        action_horizon=config.action_horizon,
        regime=config.regime,
        n_test=config.n_test,
        max_steps=config.max_steps,
        checkpoint=checkpoint,
        ckpt_slug=config.ckpt_slug,
        test_start_seed=config.test_start_seed,
    )

    result = {
        "run_dir_name": run_dir_name,
        "task": config.task,
        "policy_label": policy_label,
        "policy_source": config.policy_source,
        "checkpoint": str(checkpoint),
        "checkpoint_slug": checkpoint_slug(checkpoint, config.ckpt_slug),
        "regime": config.regime,
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
            "overlap_checkpoints": list(config.overlap_checkpoints),
        },
        "metrics": summary,
    }

    out_path = output_dir / "eval_log.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    _print_summary(summary)
    return result


def _print_summary(summary: dict[str, Any]) -> None:
    print("--- Push-T eval summary ---")
    for key in ("mean_score", "n_episodes"):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    for key in sorted(summary):
        if isinstance(key, str) and key.startswith("mean_score@"):
            print(f"  {key}: {summary[key]}")


def main() -> int:
    """CLI for local debugging (requires GPU + diffusion_policy submodule)."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=("dp_baseline", "soda"), default="dp_baseline")
    parser.add_argument("--n-test", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--regime",
        default="receding",
        choices=("receding", "open"),
        help="receding=replan after action_horizon steps; open=single plan for full chunk",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Actions executed before replanning (default: 1)",
    )
    args = parser.parse_args()

    cfg = EvalConfig(
        policy_source=args.policy,
        checkpoint_path=args.checkpoint,
        n_test=args.n_test,
        max_steps=args.max_steps,
        regime=args.regime,
        action_horizon=args.action_horizon,
    )
    run_pusht_eval(cfg, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
