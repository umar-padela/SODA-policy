"""
Generic Push-T eval orchestrator (project_plan §6, §7 rows 8–9, 17, 22).

Loads a policy, runs sim rollouts, computes overlap metrics at multiple step horizons.
Eval settings and checkpoint paths come from a yaml config file; see ``eval_yaml.py``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from soda.eval.eval_manifest import (
    build_eval_output_dir,
    config_stem_from_path,
    rename_eval_videos,
    run_dir_name_from_path,
    write_eval_run_manifest,
)
from soda.eval.eval_yaml import EvalConfig, build_eval_config_from_yaml
from soda.eval.eval_yaml import (
    EvalCliOverrides,
    finalize_eval_config,
    resolve_eval_config_path,
    _resolve_checkpoint,
)
from soda.eval.policy_loaders import load_pusht_yaml
from soda.eval.run_naming import (
    build_eval_run_dir_name,
    checkpoint_slug,
    policy_label_from_config,
)
from soda.eval.dp_runner import run_dp_pusht_eval
from soda.eval.low_only_runner import run_low_only_pusht_eval
from soda.eval.soda_runner import run_soda_pusht_eval
from soda.eval.wandb_logging import log_eval_to_wandb

# Re-export for modal_config and tests.
__all__ = [
    "EvalCliOverrides",
    "EvalConfig",
    "build_eval_config_from_yaml",
    "finalize_eval_config",
    "resolve_eval_config_path",
    "resolve_eval_output_dir",
    "run_pusht_eval",
]


def resolve_eval_output_dir(
    config: EvalConfig,
    experiments_root: Path,
    *,
    timestamp: datetime | None = None,
) -> tuple[Path, datetime]:
    """Build ``{root}/eval/{config_stem}/{YYYYMMDD}/{HHMMSS}/``."""
    return build_eval_output_dir(
        experiments_root,
        config_path=config.config_path,
        timestamp=timestamp,
    )


def run_pusht_eval(
    config: EvalConfig,
    output_dir: Path,
    *,
    experiments_root: Path | None = None,
    invoke_command: str | None = None,
    cli: EvalCliOverrides | None = None,
    run_timestamp: datetime | None = None,
) -> dict[str, Any]:
    """End-to-end eval: load checkpoint → rollouts → write JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = finalize_eval_config(config)
    checkpoint = _resolve_checkpoint(config)
    config_stem = config_stem_from_path(config.config_path)
    eval_yaml_cfg = load_pusht_yaml(config.config_path)

    write_eval_run_manifest(
        output_dir,
        config=config,
        invoke_command=invoke_command,
        cli=cli,
        timestamp=run_timestamp,
    )

    runner_log: dict[str, Any] | None = None
    if config.policy_source in ("dp_frozen", "dp"):
        summary, runner_log = run_dp_pusht_eval(
            checkpoint=checkpoint,
            output_dir=output_dir,
            device=config.device,
            eval_yaml_cfg=eval_yaml_cfg,
            n_test=config.n_test,
            n_train=config.n_train,
            n_test_vis=config.n_test_vis,
            n_action_steps=config.n_action_steps,
            max_steps=config.max_steps,
            test_start_seed=config.test_start_seed,
            train_start_seed=config.train_start_seed,
            record_video=config.record_video,
            overlap_checkpoints=config.overlap_checkpoints,
        )
    elif config.policy_source == "soda":
        summary, runner_log = run_soda_pusht_eval(
            checkpoint=checkpoint,
            output_dir=output_dir,
            device=config.device,
            config_name=config.config_name,
            n_test=config.n_test,
            n_train=config.n_train,
            n_test_vis=config.n_test_vis,
            n_action_steps=config.n_action_steps,
            max_steps=config.max_steps,
            test_start_seed=config.test_start_seed,
            train_start_seed=config.train_start_seed,
            record_video=config.record_video,
            overlap_checkpoints=config.overlap_checkpoints,
            high_checkpoint=config.high_checkpoint,
            low_checkpoint=config.low_checkpoint,
        )
    elif config.policy_source == "soda_low":
        assert config.fixed_option_id is not None
        assert config.low_checkpoint is not None
        summary, runner_log = run_low_only_pusht_eval(
            checkpoint=config.low_checkpoint,
            fixed_option_id=config.fixed_option_id,
            output_dir=output_dir,
            device=config.device,
            config_name=config.config_name,
            n_test=config.n_test,
            n_train=config.n_train,
            n_test_vis=config.n_test_vis,
            n_action_steps=config.n_action_steps,
            max_steps=config.max_steps,
            test_start_seed=config.test_start_seed,
            train_start_seed=config.train_start_seed,
            record_video=config.record_video,
            overlap_checkpoints=config.overlap_checkpoints,
        )
    else:
        raise ValueError(f"Unknown policy_source: {config.policy_source}")

    if runner_log is not None:
        renamed = rename_eval_videos(
            output_dir,
            config_stem=config_stem,
            runner_log=runner_log,
        )
        if renamed:
            summary["video_paths"] = renamed

    policy_label = policy_label_from_config(
        config.policy_source,
        config.config_name,
        fixed_option_id=config.fixed_option_id,
    )
    root = experiments_root
    if root is not None:
        run_dir_name = run_dir_name_from_path(output_dir, root)
    else:
        run_dir_name = output_dir.name
    descriptive_name = build_eval_run_dir_name(
        task=config.task,
        policy_label=policy_label,
        n_action_steps=config.n_action_steps,
        n_test=config.n_test,
        max_steps=config.max_steps,
        checkpoint=checkpoint,
        ckpt_slug=config.ckpt_slug,
        test_start_seed=config.test_start_seed,
        timestamp=run_timestamp,
    )

    result = {
        "run_dir_name": run_dir_name,
        "descriptive_run_name": descriptive_name,
        "output_dir": str(output_dir),
        "task": config.task,
        "policy_label": policy_label,
        "policy_source": config.policy_source,
        "config_path": str(config.config_path),
        "config_stem": config_stem,
        "config_name": config.config_name,
        "checkpoint": str(checkpoint),
        "checkpoint_slug": checkpoint_slug(checkpoint, config.ckpt_slug),
        "config": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(config).items()},
            "overlap_checkpoints": list(config.overlap_checkpoints),
        },
        "metrics": summary,
    }
    if runner_log is not None:
        log_key = (
            "dp_runner_log"
            if config.policy_source in ("dp_frozen", "dp")
            else "low_only_runner_log"
            if config.policy_source == "soda_low"
            else "soda_runner_log"
        )
        result[log_key] = runner_log

    out_path = output_dir / "eval_log.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")

    episodes = summary.get("episodes")
    if episodes:
        episodes_path = output_dir / "episodes.json"
        episodes_path.write_text(json.dumps(episodes, indent=2), encoding="utf-8")
        print(f"Wrote {episodes_path}")

    log_eval_to_wandb(
        result,
        enabled=config.wandb_enabled,
        project=config.wandb_project,
        run_name=descriptive_name,
    )

    _print_summary(summary)
    if summary.get("video_paths"):
        print("  videos:")
        for path in summary["video_paths"]:
            print(f"    {path}")
    return result


def _print_summary(summary: dict[str, Any]) -> None:
    print("--- Push-T eval summary ---")
    if "n_episodes" in summary:
        print(f"  n_episodes: {summary['n_episodes']}")
    if "mean_score" in summary:
        std = summary.get("std_score")
        if std is not None:
            print(f"  mean_score: {summary['mean_score']:.4f} ± {std:.4f}")
        else:
            print(f"  mean_score: {summary['mean_score']}")
    checkpoint_steps = sorted(
        int(key.split("@", 1)[1])
        for key in summary
        if isinstance(key, str) and key.startswith("mean_score@")
    )
    for step in checkpoint_steps:
        mean_key = f"mean_score@{step}"
        std_key = f"std_score@{step}"
        mean_val = summary.get(mean_key)
        std_val = summary.get(std_key)
        if std_val is not None:
            print(f"  {mean_key}: {mean_val:.4f} ± {std_val:.4f}")
        elif mean_val is not None:
            print(f"  {mean_key}: {mean_val}")
    if summary.get("episodes"):
        print(
            f"  per-episode scores: metrics.episodes in eval_log.json "
            f"({len(summary['episodes'])} episodes)"
        )


def main() -> int:
    """CLI for local debugging (requires GPU + diffusion_policy submodule)."""
    import argparse
    import shlex
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pusht/dp_frozen.yaml"),
        help="Eval yaml (checkpoint paths + eval.smoke/full settings)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir (default: auto under experiments/eval/{config_stem}/...)",
    )
    parser.add_argument("--full", action="store_true", help="Use eval.full regime from yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--high-checkpoint", type=Path, default=None)
    parser.add_argument("--low-checkpoint", type=Path, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--n-test-vis", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("dp_frozen", "dp", "soda", "soda_low"),
        default=None,
        help="Override eval.policy from yaml",
    )
    parser.add_argument(
        "--fixed-option-id",
        type=int,
        default=None,
        help="Required for soda_low: hold this skill id for the whole episode (0–2 on Push-T)",
    )
    parser.add_argument(
        "--run-readme",
        type=str,
        default=None,
        help="Required description of this eval run (also via SODA_RUN_README env)",
    )
    args = parser.parse_args()

    cli = EvalCliOverrides(
        checkpoint_path=args.checkpoint,
        high_checkpoint=args.high_checkpoint,
        low_checkpoint=args.low_checkpoint,
        full=args.full,
        n_test=args.n_test,
        max_steps=args.max_steps,
        n_action_steps=args.n_action_steps,
        n_test_vis=args.n_test_vis,
        record_video=not args.no_video,
        policy=args.policy,
        fixed_option_id=args.fixed_option_id,
        run_readme=args.run_readme,
    )
    cfg = build_eval_config_from_yaml(args.config, cli=cli)

    if args.output_dir is not None:
        out_dir = args.output_dir
        run_ts = None
    else:
        out_dir, run_ts = resolve_eval_output_dir(cfg, Path("experiments"))

    invoke_command = " ".join(shlex.quote(str(x)) for x in sys.argv)
    run_pusht_eval(
        cfg,
        out_dir,
        experiments_root=Path("experiments"),
        invoke_command=invoke_command,
        cli=cli,
        run_timestamp=run_ts,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
