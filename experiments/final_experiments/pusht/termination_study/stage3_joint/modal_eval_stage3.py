"""
Eval stage3 joint runs + compare to best frozen from stage2.

Incremental: loads existing results from Modal Volume (authoritative) or local fallback,
skips already-evaluated (run, epoch) pairs. Safe to re-run.

All collection and saving happens inside a Modal remote function, so the local terminal
can be closed after spawning. Use --detach to return immediately:

  modal run --detach experiments/final_experiments/pusht/termination_study/stage3_joint/modal_eval_stage3.py

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

try:
    sys.path.insert(0, str(Path(__file__).parents[5] / "modal"))
except IndexError:
    sys.path.insert(0, "/root/soda-policy/modal")

from modal_config import app, rollout_hierarchical, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = (
    "/experiments/pusht/train_high_conditioned_prev_option/best.ckpt"
)
N_EPISODES = 50
TEST_START_SEED = 100000

OUTPUT_PATH = Path(
    "experiments/final_experiments/pusht/termination_study/stage3_joint/stage3_results.json"
)
VOLUME_OUTPUT_PATH = (
    f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/termination_study/stage3_joint/stage3_results.json"
)

STAGE3_RUNS = {
    "bottleneck_expert_joint": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage3/bottleneck_expert_joint",
        "config": "configs/pusht/exp_term_bottleneck_expert.yaml",
    },
    "bottleneck_ddim_positive_joint": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage3/bottleneck_ddim_positive_joint",
        "config": "configs/pusht/exp_term_bottleneck_ddim_positive.yaml",
    },
    "bottleneck_ddim_positive_negative_joint": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage3/bottleneck_ddim_positive_negative_joint",
        "config": "configs/pusht/exp_term_bottleneck_ddim_positive_negative.yaml",
    },
    "obs_positive_joint": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage3/obs_positive_joint",
        "config": "configs/pusht/exp_term_obs_positive.yaml",
    },
    "obs_positive_negative_joint": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage3/obs_positive_negative_joint",
        "config": "configs/pusht/exp_term_obs_positive_negative.yaml",
    },
    "both_joint": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage3/both_joint",
        "config": "configs/pusht/exp_term_both_joint.yaml",
    },
}


def _run_plot(output_path: Path) -> None:
    plot_script = Path(__file__).parent / "plot_stage3.py"
    stage2_path = Path(__file__).parents[1] / "stage2_input_comparison" / "stage2_results.json"
    if not plot_script.exists():
        print(f"  (no plot script at {plot_script}, skipping)")
        return
    if not stage2_path.exists():
        print(f"  Skipping plot: stage2 data not found at {stage2_path}")
        return
    print(f"\nRunning {plot_script.name}...")
    subprocess.run(
        [sys.executable, str(plot_script), "--stage3-data", str(output_path), "--stage2-data", str(stage2_path)],
        check=False,
    )


def _load_existing(vol: modal.Volume) -> dict[str, list[dict]]:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        existing = d.get("results", {})
        print(f"Loaded existing results from volume: { {k: len(v) for k, v in existing.items()} }")
        return existing
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        existing = d.get("results", {})
        print(f"Loaded existing results from local: { {k: len(v) for k, v in existing.items()} }")
        return existing
    return {k: [] for k in STAGE3_RUNS}


def _already_evaluated(existing: dict[str, list[dict]]) -> set[tuple[str, int]]:
    done = set()
    for run_label, results in existing.items():
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


def _print_summary(results: dict[str, list[dict]]) -> None:
    print("\n=== Stage 3 results ===")
    for run_label, run_results in results.items():
        if not run_results:
            print(f"  {run_label}: no results yet")
            continue
        best = max(run_results, key=lambda r: r["mean_score"])
        print(f"  {run_label}: best epoch={best['epoch']}, mean={best['mean_score']:.4f} ± {best['std_score']:.4f}  ({len(run_results)} epochs evaluated)")


@app.function(
    image=image,
    gpu=None,
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_stage3(
    work_items: list[dict],
    existing: dict[str, list[dict]],
    n_action_steps: int,
) -> dict:
    """Spawn rollouts, collect results, save JSON to Modal Volume. Runs entirely on Modal."""
    import json
    import numpy as np
    from pathlib import Path

    calls = []
    for item in work_items:
        call = rollout_hierarchical.spawn(
            config_path=item["config"],
            high_checkpoint=HIGH_CHECKPOINT,
            low_checkpoint=item["ckpt_path"],
            n_episodes=N_EPISODES,
            test_start_seed=TEST_START_SEED,
            n_action_steps=n_action_steps,
            duration_termination=False,
            open_loop=False,
            max_steps=300,
            no_video=True,
        )
        calls.append((item, call))

    print(f"Spawned {len(calls)} rollout jobs. Collecting results...")

    stage3_results = {k: list(v) for k, v in existing.items()}

    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

    for item, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0
        run_label = item["run_label"]
        epoch = item["epoch"]
        stage3_results.setdefault(run_label, []).append({
            "epoch": epoch, "checkpoint": item["ckpt_path"],
            "n_action_steps": n_action_steps, "duration_termination": False,
            "n_episodes": len(scores), "mean_score": mean_score,
            "std_score": std_score, "per_episode_scores": scores,
        })
        print(f"  {run_label} epoch {epoch}: mean={mean_score:.4f} ± {std_score:.4f}")

        # Save after every result — crash-safe
        for k in stage3_results:
            stage3_results[k].sort(key=lambda r: r["epoch"])
        vol_path.write_text(json.dumps({"n_action_steps": n_action_steps, "results": stage3_results}, indent=2))
        volume.commit()

    output = {"n_action_steps": n_action_steps, "results": stage3_results}
    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    return output


@app.local_entrypoint()
def main(n_action_steps: int = 8) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    eval_epochs = [50, 100]

    stage3_results = _load_existing(vol)
    already_done = _already_evaluated(stage3_results)

    work_items = []
    for run_label, run_info in STAGE3_RUNS.items():
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
            })

    if not work_items:
        print("\nNo new checkpoints to evaluate.")
        _print_summary(stage3_results)
        _download_from_volume(vol)
        _run_plot(OUTPUT_PATH)
        return

    print(f"\nSubmitting {len(work_items)} eval jobs to Modal aggregator...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    agg_call = _eval_aggregate_stage3.spawn(work_items, stage3_results, n_action_steps)
    result = agg_call.get()

    _print_summary(result["results"])
    _download_from_volume(vol)
    _run_plot(OUTPUT_PATH)
