"""
8-eval noise robustness comparison: DP vs BID vs SODA vs SODA-BID at η=0 and η=1.

Results saved as individual JSONs per (policy, η) in noise_study/:
  dp_eta0.0.json        — vanilla DP k=5, n_action_steps=1, no noise
  dp_eta1.0.json        — vanilla DP k=5, n_action_steps=1, noise η=1.0
  bid_eta0.0.json       — BID on DP k=5, no noise
  bid_eta1.0.json       — BID on DP k=5, noise η=1.0
  soda_eta0.0.json      — SODA k=5 ep350, n_action_steps=1, no noise
  soda_eta1.0.json      — SODA k=5 ep350, n_action_steps=1, noise η=1.0
  soda_bid_eta0.0.json  — SODA-BID (BID applied to π_low), no noise
  soda_bid_eta1.0.json  — SODA-BID, noise η=1.0

All JSONs use incremental format: episodes stored by seed so results can be
extended later without overwriting. Run again to add episodes.

Usage
-----
# First time (downloads BID weak checkpoint if not present):
modal run modal/modal_download_bid.py

# Run all 8 evals (25 episodes each):
modal run experiments/final_experiments/pusht/noise_study/modal_eval_bid_comparison.py \\
  --soda-low-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k5/epoch_0350.ckpt \\
  --soda-weak-checkpoint /experiments/final_experiments/pusht/kernel_size_study/k5/epoch_0050.ckpt \\
  --soda-high-checkpoint /experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt

# Extend existing results (e.g. add 25 more episodes):
modal run ... --n-episodes 50  # runs seeds 100025-100049 if 100000-100024 done
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal
import numpy as np

try:
    sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))
except IndexError:
    # Running as a flat file inside the Modal container
    sys.path.insert(0, "/root/soda-policy/modal")

from modal_config import (  # noqa: E402
    app, eval_run, eval_bid_pusht, rollout_hierarchical,
    volume, EXPERIMENTS_MOUNT, BID_PUSHT_WEAK_CKPT,
)
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
VOLUME_STUDY_DIR = f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/noise_study"

DP_CHECKPOINT   = "/experiments/dp_baselines/pusht_image_cnn_train0/best.ckpt"
DP_CONFIG       = "configs/pusht/dp_frozen.yaml"
SODA_CONFIG     = "configs/pusht/exp_k5_no_beta.yaml"  # k5 architecture (kernel_size=5)

NOISE_LEVELS    = [0.0, 1.0]
DEFAULT_N_EPS   = 25
TEST_START_SEED = 100000


# ---------------------------------------------------------------------------
# JSON helpers — incremental format
# ---------------------------------------------------------------------------

def _result_filename(policy: str, eta: float) -> str:
    return f"{policy}_eta{eta}.json"


def _volume_path(policy: str, eta: float) -> str:
    return f"{VOLUME_STUDY_DIR}/{_result_filename(policy, eta)}"


def _local_path(policy: str, eta: float) -> Path:
    return HERE / _result_filename(policy, eta)


def _load_json(vol: modal.Volume, policy: str, eta: float) -> dict:
    """Load existing result JSON from volume or local fallback."""
    vol_rel = volume_relative_path(_volume_path(policy, eta))
    try:
        data = b"".join(vol.read_file(vol_rel))
        return json.loads(data)
    except Exception:
        pass
    local = _local_path(policy, eta)
    if local.exists():
        return json.loads(local.read_text())
    return {}


def _done_seeds(existing: dict) -> set[int]:
    """Return set of already-completed seeds from existing JSON."""
    return {ep["seed"] for ep in existing.get("episodes", [])}


def _merge_results(existing: dict, new_episodes: list[dict], meta: dict) -> dict:
    """Append new episodes to existing JSON, recompute mean/std."""
    all_eps = list(existing.get("episodes", [])) + new_episodes
    # Deduplicate by seed (keep latest)
    by_seed = {ep["seed"]: ep for ep in all_eps}
    all_eps = sorted(by_seed.values(), key=lambda e: e["seed"])
    scores = [ep["score"] for ep in all_eps]
    return {
        **meta,
        "n_episodes": len(all_eps),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "episodes": all_eps,
    }


def _save_json(vol: modal.Volume, policy: str, eta: float, data: dict) -> None:
    vol_path = Path(_volume_path(policy, eta))
    vol_path.parent.mkdir(parents=True, exist_ok=True)
    vol_path.write_text(json.dumps(data, indent=2))
    volume.commit()


def _download(vol: modal.Volume, policy: str, eta: float) -> bool:
    vol_rel = volume_relative_path(_volume_path(policy, eta))
    try:
        data = b"".join(vol.read_file(vol_rel))
        local = _local_path(policy, eta)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        print(f"  Downloaded {_result_filename(policy, eta)}")
        return True
    except Exception as e:
        print(f"  Could not download {_result_filename(policy, eta)}: {e}")
        return False


# ---------------------------------------------------------------------------
# Per-policy eval helpers — return list of {seed, score} dicts
# ---------------------------------------------------------------------------

def _episodes_from_eval_run(result: dict, start_seed: int, n: int) -> list[dict]:
    eps = result.get("metrics", {}).get("episodes", [])
    if eps:
        return [{"seed": ep["seed"], "score": ep["max_overlap_full"]} for ep in eps[:n]]
    # Fallback: parse runner_log per-seed keys
    return [
        {"seed": start_seed + i,
         "score": float(result.get("dp_runner_log", result.get("soda_runner_log", {})).get(
             f"test/sim_max_reward_{start_seed + i}", 0.0))}
        for i in range(n)
    ]


def _episodes_from_rollout_hierarchical(result: dict) -> list[dict]:
    return [
        {"seed": ep["seed"], "score": ep["metrics"].get("max_overlap_full", 0.0)}
        for ep in result.get("episodes", [])
    ]


def _episodes_from_bid(bid_result: dict) -> list[dict]:
    return [
        {"seed": TEST_START_SEED + i, "score": s}
        for i, s in enumerate(bid_result.get("per_episode_scores", []))
    ]


# ---------------------------------------------------------------------------
# Modal aggregator
# ---------------------------------------------------------------------------

@app.function(
    image=__import__("modal_config", fromlist=["image"]).image,
    gpu=None,
    timeout=86400,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _run_all_conditions(
    conditions: list[dict],
    soda_config: str,
    soda_low_checkpoint: str,
    soda_weak_checkpoint: str,
    soda_high_checkpoint: str,
    bid_weak_checkpoint: str,
    n_episodes: int,
    noise_rho: float,
    max_parallel: int = 9,
    existing_data: dict | None = None,
) -> dict[str, list[dict]]:
    """
    Spawn all eval conditions in parallel. Returns {condition_key: [episode_dicts]}.

    Each condition: {"policy": str, "eta": float, "start_seed": int, "n": int}
    """
    import json
    from pathlib import Path

    def _spawn(cond: dict) -> tuple[str, float, int, object]:
        policy = cond["policy"]
        eta = float(cond["eta"])
        n = int(cond["n"])
        start = int(cond["start_seed"])
        out_dir = f"{VOLUME_STUDY_DIR}/raw/{policy}_eta{eta}_seed{start}"

        if policy == "dp":
            call = eval_run.spawn(
                config_path=DP_CONFIG,
                run_readme=f"bid_comparison dp eta={eta}",
                checkpoint_path=DP_CHECKPOINT,
                n_action_steps=1,
                n_test=n,
                n_test_vis=n,
                noise_eta=eta,
                noise_rho=noise_rho,
                record_video=True,
                output_dir=out_dir,
            )
        elif policy == "bid":
            call = eval_bid_pusht.spawn(
                checkpoint=DP_CHECKPOINT,
                reference=bid_weak_checkpoint,
                noise=eta,
                n_test=n,
                n_samples=16,
                n_mode=3,
                decay=0.9,
                output_dir=out_dir,
                noise_rho=noise_rho,
            )
        elif policy == "soda":
            call = eval_run.spawn(
                config_path=soda_config,
                run_readme=f"bid_comparison soda eta={eta}",
                high_checkpoint=soda_high_checkpoint,
                low_checkpoint=soda_low_checkpoint,
                n_action_steps=1,
                n_test=n,
                n_test_vis=n,
                noise_eta=eta,
                noise_rho=noise_rho,
                record_video=True,
                output_dir=out_dir,
            )
        elif policy == "soda_bid":
            call = rollout_hierarchical.spawn(
                config_path=soda_config,
                high_checkpoint=soda_high_checkpoint,
                low_checkpoint=soda_low_checkpoint,
                weak_low_checkpoint=soda_weak_checkpoint,
                n_episodes=n,
                test_start_seed=start,
                n_action_steps=1,
                duration_termination=True,
                high_monitors_every_step=True,
                noise_eta=eta,
                noise_rho=noise_rho,
                no_video=False,
                video_failure_threshold=101,
                output_dir=out_dir,
            )
        else:
            raise ValueError(f"Unknown policy: {policy}")

        print(f"  Spawned: {policy} η={eta} seeds={start}+{n}")
        return policy, eta, n, call

    results: dict[str, list[dict]] = {}

    # Process in batches to cap concurrent GPU containers
    for batch_start in range(0, len(conditions), max_parallel):
        batch = conditions[batch_start:batch_start + max_parallel]
        batch_calls = [_spawn(cond) for cond in batch]

        for policy, eta, n_cond, call in batch_calls:
            key = f"{policy}_eta{eta}"
            try:
                result = call.get()
                if policy in ("dp", "soda"):
                    eps = _episodes_from_eval_run(result, TEST_START_SEED, n_cond)
                elif policy == "bid":
                    eps = _episodes_from_bid(result)
                else:
                    eps = _episodes_from_rollout_hierarchical(result)
                results[key] = eps
                scores = [e["score"] for e in eps]
                mean_s = float(np.mean(scores)) if scores else 0.0
                std_s = float(np.std(scores)) if scores else 0.0
                print(f"  {key}: mean={mean_s:.1%} ± {std_s:.1%} (n={len(eps)})")

                # Save JSON to volume immediately so results persist if client disconnects
                prior = (existing_data or {}).get(key, {})
                meta = {
                    "policy": policy,
                    "noise_eta": float(cond["eta"]),
                    "noise_rho": noise_rho,
                    "n_action_steps": 1,
                    "checkpoint": DP_CHECKPOINT if policy in ("dp", "bid") else soda_low_checkpoint,
                }
                merged = _merge_results(prior, eps, meta)
                vol_path = Path(f"{VOLUME_STUDY_DIR}/{_result_filename(policy, float(cond['eta']))}")
                vol_path.parent.mkdir(parents=True, exist_ok=True)
                vol_path.write_text(json.dumps(merged, indent=2))
                volume.commit()
                print(f"  Saved -> {vol_path}")
            except Exception as e:
                print(f"  {key}: FAILED ({e})")
                results[key] = []

    return results


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    soda_low_checkpoint: str = "/experiments/final_experiments/pusht/kernel_size_study/k5/epoch_0350.ckpt",
    soda_weak_checkpoint: str = "/experiments/weak_policy/k5_weak/epoch_0015.ckpt",
    soda_high_checkpoint: str = "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt",
    bid_weak_checkpoint: str = "/experiments/weak_policy/dp_weak/checkpoints/epoch_0015.ckpt",
    soda_config: str = SODA_CONFIG,
    n_episodes: int = DEFAULT_N_EPS,
    noise_rho: float = 0.9,
    noise_levels: str = "0.0,1.0",
    skip_dp: bool = False,
    skip_bid: bool = False,
    skip_soda: bool = False,
    skip_soda_bid: bool = False,
) -> None:
    """
    Run 8-condition noise comparison. Incremental: skips already-done seeds.

    To extend results (e.g. n_episodes=50 adds episodes 100025-100049):
      modal run ... --n-episodes 50
    """
    etas = [float(x.strip()) for x in noise_levels.split(",")]
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    policies = []
    if not skip_dp:       policies.append("dp")
    if not skip_bid:      policies.append("bid")
    if not skip_soda:     policies.append("soda")
    if not skip_soda_bid: policies.append("soda_bid")

    # Determine which (policy, eta, seed_range) combinations still need work
    conditions: list[dict] = []
    existing_data: dict[tuple[str, float], dict] = {}

    for policy in policies:
        for eta in etas:
            existing = _load_json(vol, policy, eta)
            existing_data[(policy, eta)] = existing
            done = _done_seeds(existing)
            needed = [TEST_START_SEED + i for i in range(n_episodes) if TEST_START_SEED + i not in done]
            if not needed:
                n_done = len(done)
                scores = [ep["score"] for ep in existing.get("episodes", [])]
                print(f"{policy} η={eta}: already done (n={n_done}, mean={np.mean(scores):.1%}), skipping")
                continue
            start = min(needed)
            n_needed = len(needed)
            conditions.append({"policy": policy, "eta": eta, "start_seed": start, "n": n_needed})
            print(f"{policy} η={eta}: need {n_needed} episodes (seeds {start}+)")

    if not conditions:
        print("\nAll conditions already evaluated.")
    else:
        print(f"\nSubmitting {len(conditions)} eval jobs...")
        new_results = _run_all_conditions.remote(
            conditions,
            soda_config=soda_config,
            soda_low_checkpoint=soda_low_checkpoint,
            soda_weak_checkpoint=soda_weak_checkpoint,
            soda_high_checkpoint=soda_high_checkpoint,
            bid_weak_checkpoint=bid_weak_checkpoint,
            n_episodes=n_episodes,
            noise_rho=noise_rho,
            existing_data={
                f"{p}_eta{e}": d
                for (p, e), d in existing_data.items()
            },
        )

        # Merge new episodes into existing JSONs and save
        for cond in conditions:
            policy, eta = cond["policy"], float(cond["eta"])
            key = f"{policy}_eta{eta}"
            new_eps = new_results.get(key, [])
            if not new_eps:
                continue
            meta = {
                "policy": policy,
                "noise_eta": eta,
                "noise_rho": noise_rho,
                "n_action_steps": 1,
                "checkpoint": DP_CHECKPOINT if policy in ("dp", "bid") else soda_low_checkpoint,
            }
            merged = _merge_results(existing_data[(policy, eta)], new_eps, meta)
            _save_json(vol, policy, eta, merged)
            existing_data[(policy, eta)] = merged

    # Download all results locally
    print("\nDownloading results...")
    for policy in policies:
        for eta in etas:
            _download(vol, policy, eta)

    # Print summary table
    print("\n=== Noise Comparison Summary ===")
    print(f"{'Policy':<12}  {'η':>5}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("-" * 46)
    for policy in policies:
        for eta in etas:
            data = existing_data.get((policy, eta), {})
            if data.get("n_episodes", 0) > 0:
                print(
                    f"{policy:<12}  {eta:>5.1f}  "
                    f"{data['mean_score']:>7.1%}  {data['std_score']:>7.1%}  "
                    f"{data['n_episodes']:>4}"
                )
