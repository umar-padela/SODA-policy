"""
Local entrypoint for Push-T eval on Modal (§7 rows 8–9, 17, 22).

Pass a yaml config path; checkpoint paths and eval settings come from the file.
CLI flags override yaml (see ``soda/eval/eval_yaml.py``).

Examples (repo root):

  modal run modal/modal_eval.py --run-readme "Frozen DP smoke eval" \\
    --config configs/pusht/dp_frozen.yaml

  modal run modal/modal_eval.py --run-readme "SODA smoke n=1" \\
    --config configs/pusht/soda_supervised.yaml --n-test 1
"""

from modal_config import app, eval_run, spawn_modal_function


def _build_invoke_command(
    *,
    config: str,
    run_readme: str,
    full: bool,
    checkpoint: str | None,
    high_checkpoint: str | None,
    low_checkpoint: str | None,
    n_test: int | None,
    max_steps: int | None,
    n_action_steps: int | None,
    n_test_vis: int | None,
    ckpt_slug: str | None,
    no_video: bool,
) -> str:
    parts = ["modal run modal/modal_eval.py", f'--run-readme "{run_readme}"', f"--config {config}"]
    if full:
        parts.append("--full")
    if checkpoint:
        parts.append(f"--checkpoint {checkpoint}")
    if high_checkpoint:
        parts.append(f"--high-checkpoint {high_checkpoint}")
    if low_checkpoint:
        parts.append(f"--low-checkpoint {low_checkpoint}")
    if n_test is not None:
        parts.append(f"--n-test {n_test}")
    if max_steps is not None:
        parts.append(f"--max-steps {max_steps}")
    if n_action_steps is not None:
        parts.append(f"--n-action-steps {n_action_steps}")
    if n_test_vis is not None:
        parts.append(f"--n-test-vis {n_test_vis}")
    if ckpt_slug:
        parts.append(f"--ckpt-slug {ckpt_slug}")
    if no_video:
        parts.append("--no-video")
    return " ".join(parts)


@app.local_entrypoint()
def main(
    run_readme: str,
    config: str = "configs/pusht/dp_frozen.yaml",
    full: bool = False,
    checkpoint: str | None = None,
    high_checkpoint: str | None = None,
    low_checkpoint: str | None = None,
    n_test: int | None = None,
    max_steps: int | None = None,
    n_action_steps: int | None = None,
    n_test_vis: int | None = None,
    ckpt_slug: str | None = None,
    no_video: bool = False,
):
    """
    Run Push-T eval on Modal.

    Parameters
    ----------
    run_readme
        Required short description stored as ``README.md`` in the eval output dir.
    config
        Path to eval yaml (``configs/pusht/dp_frozen.yaml``, ``soda_supervised.yaml``, …).
    full
        Use ``eval.full`` instead of ``eval.smoke`` from yaml.
    checkpoint, high_checkpoint, low_checkpoint
        Override yaml checkpoint paths.
    n_test, max_steps, n_action_steps, n_test_vis
        Override yaml eval settings.
    """
    invoke_command = _build_invoke_command(
        config=config,
        run_readme=run_readme,
        full=full,
        checkpoint=checkpoint,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
        n_test=n_test,
        max_steps=max_steps,
        n_action_steps=n_action_steps,
        n_test_vis=n_test_vis,
        ckpt_slug=ckpt_slug,
        no_video=no_video,
    )
    result = spawn_modal_function(
        eval_run,
        label="eval_run",
        config_path=config,
        run_readme=run_readme,
        full=full,
        checkpoint_path=checkpoint,
        high_checkpoint=high_checkpoint,
        low_checkpoint=low_checkpoint,
        n_test=n_test,
        max_steps=max_steps,
        n_action_steps=n_action_steps,
        n_test_vis=n_test_vis,
        ckpt_slug=ckpt_slug,
        record_video=not no_video,
        invoke_command=invoke_command,
    )

    print("\n--- eval_run result ---")
    print(f"config: {result.get('config_path', config)}")
    print(f"run_dir_name: {result.get('run_dir_name')}")
    print(f"policy: {result.get('policy_label')}")
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
    print("Done. See eval_log.json, README.md, and media/*.mp4 on Modal Volume soda-experiments.")
