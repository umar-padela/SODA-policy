"""
Eval all stage 1 runs (1a + 1b) in a single call.

Spawns all 5 rollout jobs in parallel, saves to separate JSONs for
stage1 (1a) and stage1b (1b), then downloads worst-5 debug videos and
generates separate plots for each.

Usage:
  modal run --detach \
    experiments/final_experiments/pusht/termination_study/modal_eval_stage1_all.py

Safe to re-run: already-evaluated (run, epoch) pairs are skipped.
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
    sys.path.insert(0, "/root/soda-policy/modal")

from modal_config import app, rollout_hierarchical, volume, image, EXPERIMENTS_MOUNT  # noqa: E402
from soda.experiments.paths import MODAL_VOLUME_NAME, volume_relative_path  # noqa: E402

HIGH_CHECKPOINT = "/experiments/final_experiments/pusht/high_study/high_starts_prev_opt/best.ckpt"
N_EPISODES = 50
TEST_START_SEED = 100000
N_WORST_VIDEOS = 5

STAGE1_RUNS = {
    "bottleneck_expert": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage1/bottleneck_expert",
        "config": "configs/pusht/exp_term_bottleneck_expert.yaml",
        "group": "stage1",
    },
    "bottleneck_ddim_positive": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage1/bottleneck_ddim_positive",
        "config": "configs/pusht/exp_term_bottleneck_ddim_positive.yaml",
        "group": "stage1",
    },
}

STAGE1B_RUNS = {
    "obs_positive": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage1b/obs_positive",
        "config": "configs/pusht/exp_term_obs_positive.yaml",
        "group": "stage1b",
    },
    "obs_positive_negative": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage1b/obs_positive_negative",
        "config": "configs/pusht/exp_term_obs_positive_negative.yaml",
        "group": "stage1b",
    },
    "bottleneck_ddim_positive_negative": {
        "low_dir": "/experiments/final_experiments/pusht/termination_study/stage1b/bottleneck_ddim_positive_negative",
        "config": "configs/pusht/exp_term_bottleneck_ddim_positive_negative.yaml",
        "group": "stage1b",
    },
}

STAGE1_OUTPUT = Path("experiments/final_experiments/pusht/termination_study/stage1_bottleneck_source/stage1_results.json")
STAGE1B_OUTPUT = Path("experiments/final_experiments/pusht/termination_study/stage1b_obs_signal/stage1b_results.json")
STAGE1_VOL = f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/termination_study/stage1/stage1_results.json"
STAGE1B_VOL = f"{EXPERIMENTS_MOUNT}/final_experiments/pusht/termination_study/stage1b/stage1b_results.json"

LOCAL_DEBUG_BASE = Path("experiments/final_experiments/pusht/termination_study")


def _load_existing(vol: modal.Volume, vol_path: str, local_path: Path) -> dict:
    try:
        data = b"".join(vol.read_file(volume_relative_path(vol_path)))
        return json.loads(data).get("results", {})
    except Exception:
        pass
    if local_path.exists():
        return json.loads(local_path.read_text()).get("results", {})
    return {}


def _already_done(existing: dict) -> set:
    return {(run, r["epoch"]) for run, results in existing.items() for r in results}


def _find_epoch_checkpoints(vol: modal.Volume, low_dir: str, epochs: list) -> dict:
    try:
        entries = vol.listdir(volume_relative_path(low_dir))
    except Exception:
        return {}
    found = {}
    for entry in entries:
        fname = Path(entry.path).name
        m = re.match(r"epoch_(\d{4})\.ckpt$", fname)
        if m and int(m.group(1)) in epochs:
            found[int(m.group(1))] = f"{low_dir.rstrip('/')}/{fname}"
    return found


def _download_json(vol: modal.Volume, vol_path: str, local_path: Path) -> None:
    try:
        data = b"".join(vol.read_file(volume_relative_path(vol_path)))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)
        print(f"Downloaded → {local_path}")
    except Exception as e:
        print(f"  Warning: {e}")


def _download_worst_videos(vol: modal.Volume, results: dict, stage: str, subdir: str) -> None:
    downloaded = 0
    for run_label, run_results in results.items():
        for r in run_results:
            epoch = r["epoch"]
            scores = r.get("per_episode_scores", [])
            if not scores:
                continue
            worst = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1])[:N_WORST_VIDEOS]]
            vol_base = f"final_experiments/pusht/termination_study/{stage}/debug_videos/{run_label}/epoch_{epoch:04d}"
            local_base = LOCAL_DEBUG_BASE / subdir / "debug_videos" / run_label / f"epoch_{epoch:04d}"
            local_base.mkdir(parents=True, exist_ok=True)
            for ep_idx in worst:
                seed = TEST_START_SEED + ep_idx
                fname = f"ep{ep_idx:04d}_seed{seed}.mp4"
                local_file = local_base / fname
                if local_file.exists():
                    continue
                try:
                    data = b"".join(vol.read_file(volume_relative_path(f"/experiments/{vol_base}/{fname}")))
                    local_file.write_bytes(data)
                    print(f"  {fname} ({scores[ep_idx]:.1f}%)")
                    downloaded += 1
                except Exception:
                    pass
    if downloaded:
        print(f"  {downloaded} video(s) → {LOCAL_DEBUG_BASE / subdir / 'debug_videos'}")


def _run_plot(script: Path, data: Path, extra_args: list[str] | None = None) -> None:
    if not script.exists():
        return
    print(f"Plotting {script.name}...")
    cmd = [sys.executable, str(script), "--data", str(data)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=False)


def _print_summary(label: str, results: dict) -> None:
    print(f"\n=== {label} ===")
    for run_label, run_results in results.items():
        if not run_results:
            print(f"  {run_label}: no results yet")
            continue
        best = max(run_results, key=lambda r: r["mean_score"])
        print(f"  {run_label}: best epoch={best['epoch']}, mean={best['mean_score']:.4f} ± {best['std_score']:.4f}")


@app.function(image=image, gpu=None, timeout=86400, volumes={EXPERIMENTS_MOUNT: volume})
def _eval_aggregate_all(
    work_items: list[dict],
    stage1_existing: dict,
    stage1b_existing: dict,
    n_action_steps: int,
) -> dict:
    """Spawn all stage1+stage1b rollouts in parallel, save to both JSONs. Crash-safe."""
    import json
    import numpy as np
    from pathlib import Path

    calls = []
    for item in work_items:
        stage = item["group"]
        debug_dir = (
            f"/experiments/final_experiments/pusht/termination_study/{stage}"
            f"/debug_videos/{item['run_label']}/epoch_{item['epoch']:04d}"
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
            output_dir=debug_dir,
        )
        calls.append((item, call))

    print(f"Spawned {len(calls)} rollout jobs in parallel. Collecting...")

    s1 = {k: list(v) for k, v in stage1_existing.items()}
    s1b = {k: list(v) for k, v in stage1b_existing.items()}
    s1_path = Path(STAGE1_VOL)
    s1b_path = Path(STAGE1B_VOL)
    s1_path.parent.mkdir(parents=True, exist_ok=True)
    s1b_path.parent.mkdir(parents=True, exist_ok=True)

    def _save():
        for d in (s1, s1b):
            for k in d:
                d[k].sort(key=lambda r: r["epoch"])
        s1_best = {k: max(v, key=lambda r: r["mean_score"]) for k, v in s1.items() if v}
        s1b_best = {k: max(v, key=lambda r: r["mean_score"]) for k, v in s1b.items() if v}
        s1_win = max(s1_best, key=lambda k: s1_best[k]["mean_score"]) if s1_best else None
        s1b_win = max(s1b_best, key=lambda k: s1b_best[k]["mean_score"]) if s1b_best else None
        s1_path.write_text(json.dumps({"n_action_steps": n_action_steps, "results": s1, "best_per_run": s1_best, "winner": s1_win}, indent=2))
        s1b_path.write_text(json.dumps({"n_action_steps": n_action_steps, "results": s1b, "best_per_run": s1b_best, "winner": s1b_win}, indent=2))
        volume.commit()

    for item, call in calls:
        result = call.get()
        episodes = result.get("episodes") or []
        scores = [ep["metrics"].get("max_overlap_full", 0.0) for ep in episodes]
        mean_score = float(np.mean(scores)) if scores else 0.0
        std_score = float(np.std(scores)) if scores else 0.0
        entry = {
            "epoch": item["epoch"],
            "checkpoint": item["ckpt_path"],
            "n_action_steps": n_action_steps,
            "duration_termination": False,
            "n_episodes": len(scores),
            "mean_score": mean_score,
            "std_score": std_score,
            "per_episode_scores": scores,
        }
        target = s1 if item["group"] == "stage1" else s1b
        target.setdefault(item["run_label"], []).append(entry)
        print(f"  [{item['group']}] {item['run_label']} ep{item['epoch']}: mean={mean_score:.4f} ± {std_score:.4f}")
        _save()  # crash-safe

    s1_best = {k: max(v, key=lambda r: r["mean_score"]) for k, v in s1.items() if v}
    s1b_best = {k: max(v, key=lambda r: r["mean_score"]) for k, v in s1b.items() if v}
    s1_win = max(s1_best, key=lambda k: s1_best[k]["mean_score"]) if s1_best else None
    s1b_win = max(s1b_best, key=lambda k: s1b_best[k]["mean_score"]) if s1b_best else None
    return {
        "stage1":  {"results": s1,  "best_per_run": s1_best,  "winner": s1_win},
        "stage1b": {"results": s1b, "best_per_run": s1b_best, "winner": s1b_win},
    }


@app.local_entrypoint()
def main(n_action_steps: int = 8, download_only: bool = False) -> None:
    vol = modal.Volume.from_name(MODAL_VOLUME_NAME)
    eval_epochs = [50, 100]

    s1_existing  = _load_existing(vol, STAGE1_VOL,  STAGE1_OUTPUT)
    s1b_existing = _load_existing(vol, STAGE1B_VOL, STAGE1B_OUTPUT)

    here = Path(__file__).parent
    plot1   = here / "stage1_bottleneck_source/plot_stage1.py"
    plot1b  = here / "stage1b_obs_signal/plot_stage1b.py"

    if download_only:
        _print_summary("Stage 1a", s1_existing)
        _print_summary("Stage 1b", s1b_existing)
        _download_json(vol, STAGE1_VOL,  STAGE1_OUTPUT)
        _download_json(vol, STAGE1B_VOL, STAGE1B_OUTPUT)
        _download_worst_videos(vol, s1_existing,  "stage1",  "stage1_bottleneck_source")
        _download_worst_videos(vol, s1b_existing, "stage1b", "stage1b_obs_signal")
        _run_plot(plot1,  STAGE1_OUTPUT, ["--stage1b-data", str(STAGE1B_OUTPUT)])
        _run_plot(plot1b, STAGE1B_OUTPUT)
        return

    done1  = _already_done(s1_existing)
    done1b = _already_done(s1b_existing)

    work_items = []
    for run_label, info in {**STAGE1_RUNS, **STAGE1B_RUNS}.items():
        done = done1 if info["group"] == "stage1" else done1b
        ckpts = _find_epoch_checkpoints(vol, info["low_dir"], eval_epochs)
        new = {e: p for e, p in ckpts.items() if (run_label, e) not in done}
        skipped = len(ckpts) - len(new)
        print(f"[{info['group']}] {run_label}: {len(ckpts)} found, {skipped} done, {len(new)} new")
        for epoch, ckpt_path in sorted(new.items()):
            work_items.append({"run_label": run_label, "epoch": epoch, "ckpt_path": ckpt_path,
                                "config": info["config"], "group": info["group"]})

    if not work_items:
        print("\nNo new checkpoints to evaluate.")
        _print_summary("Stage 1a", s1_existing)
        _print_summary("Stage 1b", s1b_existing)
        _download_json(vol, STAGE1_VOL,  STAGE1_OUTPUT)
        _download_json(vol, STAGE1B_VOL, STAGE1B_OUTPUT)
        _download_worst_videos(vol, s1_existing,  "stage1",  "stage1_bottleneck_source")
        _download_worst_videos(vol, s1b_existing, "stage1b", "stage1b_obs_signal")
        _run_plot(plot1,  STAGE1_OUTPUT, ["--stage1b-data", str(STAGE1B_OUTPUT)])
        _run_plot(plot1b, STAGE1B_OUTPUT)
        return

    print(f"\nSubmitting {len(work_items)} eval jobs (all in parallel)...")
    print("Safe to close terminal with --detach; results saved to Modal Volume automatically.")

    result = _eval_aggregate_all.remote(work_items, s1_existing, s1b_existing, n_action_steps)

    _print_summary("Stage 1a", result["stage1"]["results"])
    _print_summary("Stage 1b", result["stage1b"]["results"])
    if result["stage1"].get("winner"):
        print(f"\nStage 1a WINNER: {result['stage1']['winner']}")
        print("  Update BOTTLENECK_BEST_DIR/CONFIG in modal_eval_stage2.py")
    if result["stage1b"].get("winner"):
        print(f"Stage 1b WINNER: {result['stage1b']['winner']}")
        print("  Update OBS_BEST_DIR/CONFIG in modal_eval_stage2.py")
        print("  Update escape_relabeling in exp_term_both.yaml / exp_term_both_joint.yaml")

    _download_json(vol, STAGE1_VOL,  STAGE1_OUTPUT)
    _download_json(vol, STAGE1B_VOL, STAGE1B_OUTPUT)
    _download_worst_videos(vol, result["stage1"]["results"],  "stage1",  "stage1_bottleneck_source")
    _download_worst_videos(vol, result["stage1b"]["results"], "stage1b", "stage1b_obs_signal")
    _run_plot(plot1,  STAGE1_OUTPUT, ["--stage1b-data", str(STAGE1B_OUTPUT)])
    _run_plot(plot1b, STAGE1B_OUTPUT)
