"""
Eval stage2 runs + duration_termination baseline — termination_study stage 2 eval.

Incremental: loads existing results from Modal Volume (authoritative) or local fallback,
skips already-evaluated (run, epoch) pairs. Safe to re-run.

All collection and saving happens inside a Modal remote function, so the local terminal
can be closed after spawning. Use --detach to return immediately:

  modal run --detach experiments/final_experiments/pusht/termination_study/stage2_input_comparison/modal_eval_stage2.py \\
    --bottleneck-best-dir /experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert

If the terminal closes mid-run, re-running will find the completed results on the volume.
"""

from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import modal

sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))

from modal_config import app, rollout_hierarchical, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = (
    "/experiments/pusht/train_high_conditioned_prev_option/best.ckpt"
)
N_EPISODES = 50
TEST_START_SEED = 100000

OUTPUT_PATH = Path(
    "experiments/final_experiments/pusht/termination_study/stage2_input_comparison/stage2_results.json"
)
VOLUME_OUTPUT_PATH = (
    f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/termination_study/stage2_input_comparison/stage2_results.json"
)

STAGE2_CONFIGS = {
    "obs": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage2/obs",
        "config": "configs/pusht/exp_term_obs.yaml",
        "duration_termination": False,
    },
    "both": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage2/both",
        "config": "configs/pusht/exp_term_both.yaml",
        "duration_termination": False,
    },
}


def _run_plot(output_path: Path) -> None:
    plot_script = Path(__file__).parent / "plot_stage2.py"
    if not plot_script.exists():
        print(f"  (no plot script at {plot_script}, skipping)")
        return
    print(f"\nRunning {plot_script.name}...")
    subprocess.run([sys.executable, str(plot_script), "--data", str(output_path)], check=False)


def _load_existing(vol: modal.Volume) -> dict:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        print(f"Loaded existing results from volume: { {k: len(v) for k, v in d.get('results', {}).items()} }")
        return d
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        print(f"Loaded existing results from local: { {k: len(v) for k, v in d.get('results', {}).items()} }")
        return d
    return {"results": {k: [] for k in STAGE2_CONFIGS}, "duration_termination_baseline": None}


def _already_evaluated(existing: dict) -> set[tuple[str, int]]:
    done = set()
    for run_label, results in existing.get("results", {}).items():
        for r in results:
            done.add((run_label, r["epoch"]))
    return done


def _download_from_volume(vol: modal.Volume) -> None:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_bytes(data)
        print(f"Downloaded results → {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Warning: could not download from volume: {e}")


def _find_epoch_checkpoints(vol: modal.Volume, low_dir: str, epochs: list[int]) -> dict[int, str]:
    rel_dir = volume_relative_path(low_dir)
    try:
        entries = vol.listdir(rel_dir)
    except Exception:
        return {}
    found = {}
    for entry in entries:
        fname = Path(entry.path).name
        m = re.match(r"epoch_(\d{4})\.ckpt$", fname)
        if m and int(m.group(1)) in epochs:
            found[int(m.group(1))] = f"{low_dir.rstrip('/')}/{fname}"
    return found


def _print_summary(results: dict[str, list[dict]], baseline: dict | None) -> None:
    print("\n=== Stage 2 results ===")
    for run_label, run_results in results.items():
        if not run_results:
            print(f"  {run_label}: no results yet")
            continue
        best = max(run_results, key=lambda r: r["mean_score"])
        print(f"  {run_label}: best epoch={best['epoch']}, mean={best['mean_score']:.4f} ± {best['std_score']:.4f}  ({len(run_results)} epochs evaluated)")
    if baseline:
        print(f"  duration_termination baseline: mean={baseline['mean_score']:.4f}")


@app.function(
    image=image,
    gpu=None,
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_stage2(
    work_items: list[dict],
    existing: dict,
    n_action_steps: int,
    baseline_item: dict | None,
) -> dict:
    """Spawn rollouts, collect results, save JSON to Modal Volume. Runs entirely on Modal."""
    import json
    import numpy as np
    from pathlib import Path

    # Spawn all rollouts in parallel
    calls = []
    for item in work_items:
        call = rollout_hierarchical.spawn(
            config_path=item["config"],
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=item["ckpt_path"],
            n_episodes=N_EPISODES,
            test_start_seed=TEST_START_SEED,
            n_action_steps=n_action_steps,
            duration_termination=item.get("duration_termination", False),
            open_loop=False,
            max_steps=300,
            no_video=True,
        )
        calls.append((item, call))

    # Baseline (if needed)
    baseline_call = None
    if baseline_item:
        baseline_call = rollout_hierarchical.spawn(
            config_path=baseline_item["config"],
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=baseline_item["ckpt_path"],
            n_episodes=N_EPISODES,
            test_start_seed=TEST_START_SEED,
            n_action_steps=n_action_steps,
            duration_termination=True,
            open_loop=False,
            max_steps=300,
            no_video=True,
        )

    print(f"Spawned {len(calls) + (1 if baseline_call else 0)} rollout jobs. Collecting results...")

    stage2_results = {k: list(v) for k, v in existing.get("results", {}).items()}
    baseline_result = existing.get("duration_termination_baseline")

    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

    def _save():
        for k in stage2_results:
            stage2_results[k].sort(key=lambda r: r["epoch"])
        vol_path.write_text(json.dumps({
            "n_action_steps": n_action_steps,
            "results": stage2_results,
            "duration_termination_baseline": baseline_result,
        }, indent=2))
        volume.commit()

    for item, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0
        run_label = item["run_label"]
        epoch = item["epoch"]
        stage2_results.setdefault(run_label, []).append({
            "epoch": epoch, "checkpoint": item["ckpt_path"],
            "n_action_steps": n_action_steps, "duration_termination": item.get("duration_termination", False),
            "n_episodes": len(scores), "mean_score": mean_score,
            "std_score": std_score, "per_episode_scores": scores,
        })
        print(f"  {run_label} epoch {epoch}: mean={mean_score:.4f} ± {std_score:.4f}")
        _save()  # crash-safe

    if baseline_call is not None:
        result = baseline_call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        baseline_result = {
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "n_episodes": len(scores),
        }
        print(f"  duration_termination baseline: mean={baseline_result['mean_score']:.4f}")
        _save()  # crash-safe

    output = {"n_action_steps": n_action_steps, "results": stage2_results, "duration_termination_baseline": baseline_result}
    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    return output


@app.local_entrypoint()
def main(
    n_action_steps: int = 8,
    bottleneck_best_dir: str = "/experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert",
    bottleneck_best_config: str = "configs/pusht/exp_term_bottleneck_expert.yaml",
    duration_baseline_ckpt: str | None = None,
) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    eval_epochs = [50, 100]

    existing = _load_existing(vol)
    already_done = _already_evaluated(existing)

    stage2_runs = dict(STAGE2_CONFIGS)
    stage2_runs["bottleneck_best"] = {
        "low_dir": bottleneck_best_dir,
        "config": bottleneck_best_config,
        "duration_termination": False,
    }

    work_items = []
    for run_label, run_info in stage2_runs.items():
        epoch_ckpts = _find_epoch_checkpoints(vol, run_info["low_dir"], eval_epochs)
        new = [(e, p) for e, p in sorted(epoch_ckpts.items()) if (run_label, e) not in already_done]
        skipped = len(epoch_ckpts) - len(new)
        print(f"{run_label}: found epochs {sorted(epoch_ckpts.keys())}, {skipped} already done, {len(new)} new")
        for epoch, ckpt_path in new:
            work_items.append({
                "run_label": run_label,
                "epoch": epoch,
                "ckpt_path": ckpt_path,
                "config": run_info["config"],
                "duration_termination": run_info["duration_termination"],
            })

    baseline_item = None
    if duration_baseline_ckpt and existing.get("duration_termination_baseline") is None:
        baseline_item = {
            "ckpt_path": duration_baseline_ckpt,
            "config": "configs/pusht/exp_k5_no_beta.yaml",
        }
    elif duration_baseline_ckpt:
        print("duration_termination baseline: already evaluated, skipping")

    if not work_items and baseline_item is None:
        print("\nNo new checkpoints to evaluate.")
        _print_summary(existing.get("results", {}), existing.get("duration_termination_baseline"))
        _download_from_volume(vol)
        _run_plot(OUTPUT_PATH)
        return

    print(f"\nSubmitting {len(work_items) + (1 if baseline_item else 0)} eval jobs to Modal aggregator...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    agg_call = _eval_aggregate_stage2.spawn(work_items, existing, n_action_steps, baseline_item)
    result = agg_call.get()

    _print_summary(result["results"], result.get("duration_termination_baseline"))
    _download_from_volume(vol)
    _run_plot(OUTPUT_PATH)
