"""
Sweep n_action_steps on the best SODA policy from termination_study — receding_horizon_study.

Incremental: loads existing results from Modal Volume (authoritative) or local fallback,
skips already-evaluated n_action_steps values. Safe to re-run.

All collection and saving happens inside a Modal remote function, so the local terminal
can be closed after spawning. Use --detach to return immediately:

  modal run --detach experiments/final_experiments/pusht/receding_horizon_study/modal_eval_n_action_steps.py \\
    --low-checkpoint /experiments/... --config configs/pusht/...

If the terminal closes mid-run, re-running will find the completed results on the volume.
"""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import modal

sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))

from modal_config import app, rollout_hierarchical, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = (
    "/experiments/pusht/train_high_conditioned_prev_option/best.ckpt"
)
N_EPISODES = 50
TEST_START_SEED = 100000
N_ACTION_STEPS_VALUES = [2, 4, 6, 8, 10, 16]

OUTPUT_PATH = Path(
    "experiments/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/sweep_results.json"
)
VOLUME_OUTPUT_PATH = (
    f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/receding_horizon_study/n_action_steps_sweep/sweep_results.json"
)


def _run_plot(output_path: Path) -> None:
    plot_script = Path(__file__).parent / "plot_n_action_steps.py"
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
        print(f"Loaded existing results from volume: {list(d.keys())}")
        return d
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        print(f"Loaded existing results from local: {list(d.keys())}")
        return d
    return {}


def _download_from_volume(vol: modal.Volume) -> None:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_bytes(data)
        print(f"Downloaded results → {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Warning: could not download from volume: {e}")


def _print_summary(sweep_results: dict) -> None:
    if not sweep_results:
        return
    print("\n=== n_action_steps sweep summary ===")
    print(f"{'Label':>12}  {'n_steps':>8}  {'Mean':>8}  {'Std':>6}")
    print("-" * 45)
    for label_key in [f"n{n}" for n in N_ACTION_STEPS_VALUES] + ["open_loop"]:
        r = sweep_results.get(label_key)
        if r is None:
            continue
        print(f"  {label_key:>10}  {str(r.get('n_action_steps', '-')):>8}  "
              f"{r.get('mean_score', 0):.4f}  {r.get('std_score', 0):.4f}")
    if sweep_results:
        best_label = max(sweep_results, key=lambda k: sweep_results[k]["mean_score"])
        print(f"\nBest: {best_label} → mean={sweep_results[best_label]['mean_score']:.4f}")


@app.function(
    image=image,
    gpu=None,
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_n_action_steps(
    work_items: list[dict],
    existing: dict,
    low_checkpoint: str,
    config: str,
    n_episodes: int,
) -> dict:
    """Spawn rollouts for each n_action_steps, collect results, save to volume."""
    import json
    import numpy as np
    from pathlib import Path

    calls = []
    for item in work_items:
        if item["label"] == "open_loop":
            call = rollout_hierarchical.spawn(
                config_path=config,
                high_checkpoint=HIGH_CHECKPOINT,
                low_checkpoint=low_checkpoint,
                n_episodes=n_episodes,
                test_start_seed=TEST_START_SEED,
                n_action_steps=8,
                duration_termination=False,
                open_loop=True,
                max_steps=300,
                no_video=True,
            )
        else:
            call = rollout_hierarchical.spawn(
                config_path=config,
                high_checkpoint=HIGH_CHECKPOINT,
                low_checkpoint=low_checkpoint,
                n_episodes=n_episodes,
                test_start_seed=TEST_START_SEED,
                n_action_steps=item["n_action_steps"],
                duration_termination=True,
                open_loop=False,
                max_steps=300,
                no_video=True,
            )
        calls.append((item, call))

    print(f"Spawned {len(calls)} rollout jobs. Collecting results...")

    sweep_results = dict(existing)

    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

    for item, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0
        label = item["label"]
        sweep_results[label] = {
            "n_action_steps": item.get("n_action_steps"),
            "label": label,
            "mean_score": mean_score,
            "std_score": std_score,
            "n_episodes": len(scores),
            "per_episode_scores": scores,
            "low_checkpoint": low_checkpoint,
        }
        print(f"  {label:>10}: mean={mean_score:.4f}  std={std_score:.4f}")

        # Save after every result — crash-safe
        vol_path.write_text(json.dumps(sweep_results, indent=2))
        volume.commit()

    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    return sweep_results


@app.local_entrypoint()
def main(
    low_checkpoint: str,
    config: str,
    n_episodes: int = N_EPISODES,
) -> None:
    """
    Sweep n_action_steps on the best SODA checkpoint from termination_study.

    Parameters
    ----------
    low_checkpoint
        Volume path to the best low policy checkpoint from termination_study.
    config
        Config yaml matching the checkpoint.
    n_episodes
        Episodes per eval (default 50).
    """
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    sweep_results = _load_existing(vol)

    print(f"Sweeping n_action_steps on: {low_checkpoint}")
    print(f"Config: {config}")

    work_items = []
    for n in N_ACTION_STEPS_VALUES:
        label = f"n{n}"
        if label in sweep_results:
            print(f"  {label}: already evaluated, skipping")
        else:
            work_items.append({"label": label, "n_action_steps": n})

    if "open_loop" in sweep_results:
        print("  open_loop: already evaluated, skipping")
    else:
        work_items.append({"label": "open_loop", "n_action_steps": None})

    if not work_items:
        print("\nNo new configs to evaluate.")
        _print_summary(sweep_results)
        _download_from_volume(vol)
        _run_plot(OUTPUT_PATH)
        return

    print(f"\nSubmitting {len(work_items)} eval jobs to Modal aggregator...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    agg_call = _eval_aggregate_n_action_steps.spawn(work_items, sweep_results, low_checkpoint, config, n_episodes)
    result = agg_call.get()

    _print_summary(result)
    _download_from_volume(vol)
    _run_plot(OUTPUT_PATH)
