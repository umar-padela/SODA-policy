"""
Final eval: SODA vs DP baseline on Block Push.

  modal run --detach experiments/final_experiments/block_push/modal_eval_block_push.py \\
      --soda-low-checkpoint  /experiments/train_low/pusht_block_push_soda_supervised/best.ckpt \\
      --soda-high-checkpoint /experiments/train_high/pusht_block_push_soda_supervised/best.ckpt \\
      --dp-checkpoint        /experiments/train_low/pusht_block_push_dp_baseline/best.ckpt

Results saved to:
  experiments/final_experiments/block_push/results.json
  /experiments/final_experiments/block_push/results.json  (Modal volume)
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import modal

sys.path.insert(0, str(Path(__file__).parents[3] / "modal"))
from modal_config import (  # noqa: E402
    app, rollout_hierarchical, eval_run, volume, image, EXPERIMENTS_MOUNT,
)

N_EPISODES      = 50
TEST_START_SEED = 100000
OUTPUT_LOCAL    = Path("experiments/final_experiments/block_push/results.json")
OUTPUT_VOLUME   = f"{EXPERIMENTS_MOUNT}/final_experiments/block_push/results.json"
TASK            = "block_push"
CONFIG_NAME     = "soda_supervised"


def _load_existing() -> dict:
    try:
        data = b"".join(volume.read_file(OUTPUT_VOLUME.replace(f"{EXPERIMENTS_MOUNT}/", "")))
        return json.loads(data)
    except Exception:
        pass
    if OUTPUT_LOCAL.exists():
        return json.loads(OUTPUT_LOCAL.read_text())
    return {}


@app.local_entrypoint()
def main(
    soda_low_checkpoint: str  = None,
    soda_high_checkpoint: str = None,
    dp_checkpoint: str        = None,
) -> None:
    results = _load_existing()

    # ── SODA eval ─────────────────────────────────────────────────────────────
    if soda_low_checkpoint and soda_high_checkpoint and "soda" not in results:
        print(f"Running SODA eval ({N_EPISODES} episodes)...")
        scores = rollout_hierarchical.remote(
            low_checkpoint  = soda_low_checkpoint,
            high_checkpoint = soda_high_checkpoint,
            config_name     = CONFIG_NAME,
            task            = TASK,
            n_episodes      = N_EPISODES,
            test_start_seed = TEST_START_SEED,
        )
        results["soda"] = {
            "low_checkpoint":  soda_low_checkpoint,
            "high_checkpoint": soda_high_checkpoint,
            "n_episodes":      N_EPISODES,
            "mean_score":      float(np.mean(scores)),
            "std_score":       float(np.std(scores)),
            "per_episode_scores": [float(s) for s in scores],
        }
        print(f"SODA: {results['soda']['mean_score']:.3f} ± {results['soda']['std_score']:.3f}")
    elif "soda" in results:
        print(f"SODA already evaluated: {results['soda']['mean_score']:.3f}")

    # ── DP baseline eval ──────────────────────────────────────────────────────
    if dp_checkpoint and "dp_baseline" not in results:
        print(f"Running DP baseline eval ({N_EPISODES} episodes)...")
        scores = eval_run.remote(
            checkpoint      = dp_checkpoint,
            config_name     = CONFIG_NAME,
            task            = TASK,
            n_episodes      = N_EPISODES,
            test_start_seed = TEST_START_SEED,
        )
        results["dp_baseline"] = {
            "checkpoint":        dp_checkpoint,
            "n_episodes":        N_EPISODES,
            "mean_score":        float(np.mean(scores)),
            "std_score":         float(np.std(scores)),
            "per_episode_scores":[float(s) for s in scores],
        }
        print(f"DP baseline: {results['dp_baseline']['mean_score']:.3f} ± {results['dp_baseline']['std_score']:.3f}")
    elif "dp_baseline" in results:
        print(f"DP baseline already evaluated: {results['dp_baseline']['mean_score']:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_LOCAL.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUTPUT_LOCAL}")

    if results:
        print("\n=== Summary ===")
        for key, v in results.items():
            print(f"  {key:15s}: {v['mean_score']:.3f} ± {v['std_score']:.3f} (n={v['n_episodes']})")
