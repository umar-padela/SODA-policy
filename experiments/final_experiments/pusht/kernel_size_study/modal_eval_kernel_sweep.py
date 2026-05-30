"""
Epoch sweep eval for k=5, k=7, k=9 — kernel_size_study step 2.

Incremental: loads existing kernel_sweep_results.json from Modal Volume (authoritative)
or local fallback, skips already-evaluated (kernel, epoch) pairs. Safe to re-run.

All collection and saving happens inside a Modal remote function (_eval_aggregate_kernel),
so the local terminal can be closed after spawning. Use --detach to return immediately:

  modal run --detach experiments/final_experiments/pusht/kernel_size_study/modal_eval_kernel_sweep.py

Results are saved to the Modal Volume and then downloaded locally. If the terminal closes
mid-run, re-running will find the completed results on the volume automatically.

n_action_steps=8 is fixed across kernel_size_study and termination_study for
apples-to-apples comparison; it is tuned in receding_horizon_study after termination_study.
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
    sys.path.insert(0, str(Path(__file__).parents[4] / "modal"))
except IndexError:
    # Running inside a Modal container: __file__ is a flat path, not in repo tree.
    # Repo is mounted at /root/soda-policy; modal_config.py is at modal/modal_config.py.
    sys.path.insert(0, "/root/soda-policy/modal")

from modal_config import app, rollout_hierarchical, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = (
    "/experiments/pusht/train_high_conditioned_prev_option/best.ckpt"
)
N_EPISODES = 50
TEST_START_SEED = 100000
EPOCH_STRIDE = 50
MIN_EPOCH = 50

OUTPUT_PATH = Path("experiments/final_experiments/pusht/kernel_size_study/kernel_sweep_results.json")
VOLUME_OUTPUT_PATH = f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/kernel_size_study/kernel_sweep_results.json"

KERNEL_DIRS = {
    "k5": "/experiments/final_experiments/pusht/kernel_size_study/k5",
    "k7": "/experiments/final_experiments/pusht/kernel_size_study/k7",
    "k9": "/experiments/final_experiments/pusht/kernel_size_study/k9",
}

KERNEL_CONFIGS = {
    "k5": "configs/pusht/exp_k5_no_beta.yaml",
    "k7": "configs/pusht/exp_k7_no_beta.yaml",
    "k9": "configs/pusht/exp_k9_no_beta.yaml",
}


def _run_plot(output_path: Path) -> None:
    plot_script = Path(__file__).parent / "plot_kernel_study.py"
    if not plot_script.exists():
        print(f"  (no plot script at {plot_script}, skipping)")
        return
    print(f"\nRunning {plot_script.name}...")
    subprocess.run([sys.executable, str(plot_script), "--data", str(output_path)], check=False)


def _load_existing(vol: modal.Volume) -> dict[str, list[dict]]:
    """Load from Modal Volume (authoritative), falling back to local file."""
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        d = json.loads(data)
        existing = d.get("kernels", {})
        print(f"Loaded existing results from volume: { {k: len(v) for k, v in existing.items()} }")
        return existing
    except Exception:
        pass
    if OUTPUT_PATH.exists():
        d = json.loads(OUTPUT_PATH.read_text())
        existing = d.get("kernels", {})
        print(f"Loaded existing results from local: { {k: len(v) for k, v in existing.items()} }")
        return existing
    return {k: [] for k in KERNEL_DIRS}


def _already_evaluated(existing: dict[str, list[dict]]) -> set[tuple[str, int]]:
    done = set()
    for k_label, results in existing.items():
        for r in results:
            done.add((k_label, r["epoch"]))
    return done


def _download_from_volume(vol: modal.Volume) -> None:
    """Download results JSON from Modal Volume to local path."""
    vol_rel = volume_relative_path(VOLUME_OUTPUT_PATH)
    try:
        data = b"".join(vol.read_file(vol_rel))
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_bytes(data)
        print(f"Downloaded results → {OUTPUT_PATH}")
    except Exception as e:
        print(f"  Warning: could not download from volume: {e}")


def _discover_epoch_checkpoints(vol: modal.Volume, low_dir: str) -> list[tuple[int, str]]:
    """Return sorted (epoch, abs_path) for epoch_NNNN.ckpt files."""
    rel_dir = volume_relative_path(low_dir)
    try:
        entries = vol.listdir(rel_dir)
    except Exception as e:
        print(f"  Warning: could not list {low_dir}: {e}")
        return []
    results = []
    for entry in entries:
        fname = Path(entry.path).name
        m = re.match(r"epoch_(\d{4})\.ckpt$", fname)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch >= MIN_EPOCH and epoch % EPOCH_STRIDE == 0:
            results.append((epoch, f"{low_dir.rstrip('/')}/{fname}"))
    return sorted(results)


def _print_summary(kernel_results: dict[str, list[dict]]) -> None:
    print("\n=== Kernel size study summary (best epoch per kernel) ===")
    for k_label, results in kernel_results.items():
        if not results:
            print(f"  {k_label}: no results yet")
            continue
        best = max(results, key=lambda r: r["mean_score"])
        print(f"  {k_label}: best epoch={best['epoch']}, mean={best['mean_score']:.4f} ± {best['std_score']:.4f}  ({len(results)} epochs evaluated)")


@app.function(
    image=image,
    gpu=None,
    timeout=7200,
    volumes={EXPERIMENTS_MOUNT: volume},
)
def _eval_aggregate_kernel(
    work_items: list[dict],
    existing: dict[str, list[dict]],
    n_action_steps: int,
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
            duration_termination=True,
            open_loop=False,
            max_steps=300,
            no_video=True,
        )
        calls.append((item, call))

    print(f"Spawned {len(calls)} rollout jobs. Collecting results...")

    kernel_results = {k: list(v) for k, v in existing.items()}

    vol_path = Path(VOLUME_OUTPUT_PATH)
    vol_path.parent.mkdir(parents=True, exist_ok=True)

    for item, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0

        step_means: dict[str, float] = {}
        step_stds: dict[str, float] = {}
        if episodes:
            all_step_keys = {
                k for ep in episodes
                for k in ep["metrics"].get("max_overlap_at_step", {}).keys()
            }
            for t in sorted(all_step_keys, key=int):
                vals = [ep["metrics"].get("max_overlap_at_step", {}).get(t, 0.0) for ep in episodes]
                step_means[f"mean_score@{t}"] = float(np.mean(vals))
                step_stds[f"std_score@{t}"] = float(np.std(vals))

        k_label = item["run_label"]
        epoch = item["epoch"]
        kernel_results.setdefault(k_label, []).append({
            "epoch": epoch,
            "checkpoint": item["ckpt_path"],
            "n_action_steps": n_action_steps,
            "duration_termination": True,
            "n_episodes": len(scores),
            "mean_score": mean_score,
            "std_score": std_score,
            **step_means,
            **step_stds,
            "per_episode_scores": scores,
        })
        print(f"  {k_label} epoch {epoch:4d}: mean={mean_score:.4f}  std={std_score:.4f}")

        # Save after every result — crash-safe
        for k in kernel_results:
            kernel_results[k].sort(key=lambda r: r["epoch"])
        vol_path.write_text(json.dumps({"n_action_steps": n_action_steps, "kernels": kernel_results}, indent=2))
        volume.commit()

    output = {"n_action_steps": n_action_steps, "kernels": kernel_results}
    print(f"Saved to volume → {VOLUME_OUTPUT_PATH}")
    return output


@app.local_entrypoint()
def main(n_action_steps: int = 8) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)

    kernel_results = _load_existing(vol)
    already_done = _already_evaluated(kernel_results)

    work_items = []
    for k_label, low_dir in KERNEL_DIRS.items():
        epoch_ckpts = _discover_epoch_checkpoints(vol, low_dir)
        new = [(e, p) for e, p in epoch_ckpts if (k_label, e) not in already_done]
        skipped = len(epoch_ckpts) - len(new)
        print(f"{k_label}: {len(epoch_ckpts)} checkpoints found, {skipped} already done, {len(new)} new")
        for epoch, ckpt_path in new:
            work_items.append({
                "run_label": k_label,
                "epoch": epoch,
                "ckpt_path": ckpt_path,
                "config": KERNEL_CONFIGS[k_label],
            })

    if not work_items:
        print("\nNo new checkpoints to evaluate.")
        _print_summary(kernel_results)
        _download_from_volume(vol)
        _run_plot(OUTPUT_PATH)
        return

    print(f"\nSubmitting {len(work_items)} eval jobs to Modal aggregator...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    agg_call = _eval_aggregate_kernel.spawn(work_items, kernel_results, n_action_steps)
    result = agg_call.get()

    _print_summary(result["kernels"])
    _download_from_volume(vol)
    _run_plot(OUTPUT_PATH)
