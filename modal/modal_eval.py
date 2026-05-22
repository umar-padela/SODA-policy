"""
Local entrypoint for Push-T eval on Modal (§7 rows 8–9, 17, 22).

DP baselines use Columbia ``PushTImageRunner`` (via ``soda.eval.dp_runner``) with
``n_action_steps=1`` by default (closed-loop). MP4s land in ``output_dir/media/``.
Every run requires ``--checkpoint`` (frozen DP, your DP retrain, or SODA).

Examples (repo root):

  # One-time: download Columbia frozen DP to the Volume
  modal run modal/modal_download_dp.py

  # §7 row 8 — smoke eval (5 episodes) with frozen DP
  modal run modal/modal_eval.py --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt

  # Quick wiring check (1 episode × 5 steps)
  modal run modal/modal_eval.py --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt --n-test 1 --max-steps 5

  # Full baseline (50 episodes)
  modal run modal/modal_eval.py --checkpoint /experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt --full

  # Your own DP or SODA checkpoint
  modal run modal/modal_eval.py --policy soda --checkpoint /experiments/train_high/.../epoch.ckpt
"""

from modal_config import FROZEN_DP_PUSHT_CHECKPOINT, app, eval_run


@app.local_entrypoint()
def main(
    checkpoint: str,
    policy: str = "dp_baseline",
    task: str = "pusht",
    config_name: str = "soda_supervised",
    ckpt_slug: str | None = None,
    full: bool = False,
    regime: str = "receding",
    action_horizon: int = 1,
    n_test: int | None = None,
    max_steps: int = 300,
    record_video: bool = True,
    n_test_vis: int | None = None,
    no_video: bool = False,
):
    """
    Run Push-T eval on Modal.

    Parameters
    ----------
    checkpoint
        Required path to ``.ckpt`` on the Modal Volume. Frozen Columbia DP:
        ``/experiments/dp_baselines/pusht_image_cnn_train0/latest.ckpt``
        (see ``modal run modal/modal_download_dp.py``).
    policy
        ``dp_baseline`` → ``diffusion_policy`` in run folder name; ``soda`` uses
        ``config_name`` (``soda_supervised`` / ``soda_unsupervised``).
    task
        ``pusht`` or ``square`` (eval code is Push-T only for now).
    config_name
        Hydra config for SODA (``soda_supervised`` or ``soda_unsupervised``).
    ckpt_slug
        Optional short checkpoint label if auto-slug from path is unclear.
    full
        If set, use 50 test episodes; default smoke uses 5.
    regime
        ``receding`` (default): replan every ``action_horizon`` steps (default 1).
        ``open``: single plan, no replan.
    action_horizon
        DP ``n_action_steps``: actions executed per replan (default 1 = closed-loop).
    n_test
        Override episode count.
    max_steps
        Max sim steps per episode (use 5 for a fast wiring check).
    record_video
        Save MP4 rollouts for the first ``n_test_vis`` test episodes.
    n_test_vis
        How many test episodes get videos (default ``min(n_test, 4)``).
    no_video
        Skip MP4 recording.
    """
    result = eval_run.remote(
        checkpoint_path=checkpoint,
        policy_source=policy,
        task=task,
        soda_config_name=config_name,
        ckpt_slug=ckpt_slug,
        smoke=not full,
        regime=regime,
        action_horizon=action_horizon,
        n_test=n_test,
        max_steps=max_steps,
        record_video=record_video and not no_video,
        n_test_vis=n_test_vis,
    )

    print("\n--- eval_run result ---")
    print(f"run_dir_name: {result.get('run_dir_name')}")
    print(f"policy: {result.get('policy_label', policy)}")
    print(f"checkpoint: {result.get('checkpoint')}")
    print(f"output_dir: {result.get('output_dir')}")
    metrics = (result.get("metrics") or {})
    for key in ("mean_score", "n_episodes"):
        if key in metrics:
            print(f"  {key}: {metrics[key]}")
    for key in sorted(metrics):
        if isinstance(key, str) and key.startswith("mean_score@"):
            print(f"  {key}: {metrics[key]}")
    videos = (metrics.get("video_paths") or [])
    if videos:
        print("  video_paths:")
        for path in videos:
            print(f"    {path}")
    print("Done. See eval_log.json and media/*.mp4 on Modal Volume soda-experiments.")


# Documented default for frozen DP (still requires explicit --checkpoint at CLI).
DEFAULT_FROZEN_DP_CHECKPOINT = FROZEN_DP_PUSHT_CHECKPOINT
