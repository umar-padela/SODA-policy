"""
Eval bottleneck_expert_positive_negative_separate — termination_study stage 1 extension.

Evaluates checkpoints at epochs [10, 20, ..., 250]. Incremental: skips already-evaluated
epochs. Safe to re-run as new checkpoints arrive.

Usage (repo root):
  modal run --detach \\
    experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/modal_eval_stage1_separate.py
"""

from __future__ import annotations
import json
import re
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
    "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt"
)
N_EPISODES = 50
TEST_START_SEED = 100000
EVAL_EPOCHS = list(range(50, 251, 50))  # [50, 100, 150, 200, 250]

RUN_LABEL = "bottleneck_expert_positive_negative_separate"
LOW_DIR = f"/experiments/final_experiments/pusht/termination_study/stage1/{RUN_LABEL}"
CONFIG = "configs/pusht/exp_term_bottleneck_expert_positive_negative_separate.yaml"

OUTPUT_PATH = Path(
    "experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/stage1_separate_results.json"
)
VOLUME_OUTPUT_PATH = (
    f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/termination_study/stage1/stage1_separate_results.json"
)


def _load_existing(vol: modal.Volume) -> list[dict]:
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        return json.loads(data).get("results", [])
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        return json.loads(OUTPUT_PATH.read_text()).get("results", [])
    return []


def _find_epoch_checkpoints(vol: modal.Volume, epochs: list[int]) -> dict[int, str]:
    rel_dir = volume_relative_path(LOW_DIR)
    try:
        entries = vol.listdir(rel_dir)
    except Exception as e:
        print(f"  Warning: {LOW_DIR}: {e}")
        return {}
    found = {}
    for entry in entries:
        fname = Path(entry.path).name
        m = re.match(r"epoch_(\d{4})\.ckpt$", fname)
        if m and int(m.group(1)) in epochs:
            found[int(m.group(1))] = f"{LOW_DIR.rstrip('/')}/{fname}"
    return found


def _print_summary(results: list[dict]) -> None:
    print(f"\n=== {RUN_LABEL} results ===")
    if not results:
        print("  no results yet")
        return
    for r in sorted(results, key=lambda r: r["epoch"]):
        print(f"  epoch={r['epoch']}: mean={r['mean_score']:.4f} ± {r['std_score']:.4f}")
    best = max(results, key=lambda r: r["mean_score"])
    print(f"  BEST: epoch={best['epoch']}, mean={best['mean_score']:.4f}")


@app.function(
    image=image,
    gpu=None,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate(
    work_items: list[dict],
    existing: list[dict],
    n_action_steps: int,
) -> dict:
    import json
    import numpy as np
    from pathlib import Path

    # Spawn all rollout jobs in parallel
    calls = []
    for item in work_items:
        debug_video_dir = (
            f"/experiments/final_experiments/pusht/termination_study/stage1"
            f"/debug_videos/{RUN_LABEL}/epoch_{item['epoch']:04d}"
        )
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
            no_video=False,
            output_dir=debug_video_dir,
        )
        calls.append((item, call))

    print(f"Spawned {len(calls)} rollout jobs in parallel. Collecting...")
    results = list(existing)
    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

    for item, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0
        results.append({
            "epoch": item["epoch"],
            "checkpoint": item["ckpt_path"],
            "n_action_steps": n_action_steps,
            "n_episodes": len(scores),
            "mean_score": mean_score,
            "std_score": std_score,
            "per_episode_scores": scores,
        })
        print(f"  epoch {item['epoch']}: mean={mean_score:.4f} ± {std_score:.4f}")
        results.sort(key=lambda r: r["epoch"])
        best = max(results, key=lambda r: r["mean_score"])
        vol_path.write_text(json.dumps({
            "run": RUN_LABEL,
            "n_action_steps": n_action_steps,
            "results": results,
            "best": best,
        }, indent=2))
        volume.commit()

    best = max(results, key=lambda r: r["mean_score"])
    return {"results": results, "best": best}


@app.local_entrypoint()
def main(n_action_steps: int = 8) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    existing = _load_existing(vol)
    done_epochs = {r["epoch"] for r in existing}

    epoch_ckpts = _find_epoch_checkpoints(vol, EVAL_EPOCHS)
    new = [(e, p) for e, p in sorted(epoch_ckpts.items()) if e not in done_epochs]
    skipped = len(epoch_ckpts) - len(new)
    print(f"{RUN_LABEL}: found epochs {sorted(epoch_ckpts.keys())}, {skipped} already done, {len(new)} new")

    if not new:
        print("No new checkpoints to evaluate.")
        _print_summary(existing)
        return

    work_items = [
        {"epoch": e, "ckpt_path": p, "config": CONFIG}
        for e, p in new
    ]

    print(f"\nSubmitting {len(work_items)} eval jobs (all parallel on Modal)...")
    agg_call = _eval_aggregate.spawn(work_items, existing, n_action_steps)
    result = agg_call.get()

    _print_summary(result["results"])
    best = result["best"]
    print(f"\nBEST: epoch={best['epoch']}, mean={best['mean_score']:.4f}")

    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_bytes(data)
        print(f"Downloaded results → {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Warning: could not download from volume: {e}")
