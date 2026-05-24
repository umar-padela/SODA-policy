"""
Local entrypoint — train vanilla Diffusion Policy on Modal.

  modal run modal/modal_train_dp.py --run-readme "Vanilla DP baseline train"
  modal run modal/modal_train_dp.py --config-name dp --task pusht --run-readme "..."
  modal run modal/modal_train_dp.py --hydra-overrides "train_dp.num_epochs=10" --run-readme "..."
  modal run modal/modal_train_dp.py --detach --run-readme "..."

Remote: ``modal_config.train_dp`` → ``python soda/training/train_dp.py`` on GPU.
Canonical checkpoints: ``/experiments/train_dp/{config_name}/`` on Volume ``soda-experiments``.
Archived runs: ``.../{config_name}/runs/{YYYYMMDD_HHMMSS}/``.
"""

from modal_config import app, spawn_modal_function, train_dp


def _build_invoke_command(
    *,
    config_name: str,
    task: str,
    detach: bool,
    hydra_overrides: str,
    run_readme: str,
) -> str:
    parts = [
        "modal run modal/modal_train_dp.py",
        f"--config-name {config_name}",
        f"--task {task}",
    ]
    if hydra_overrides.strip():
        parts.append(f'--hydra-overrides "{hydra_overrides.strip()}"')
    if detach:
        parts.append("--detach")
    parts.append(f'--run-readme "{run_readme}"')
    return " ".join(parts)


@app.local_entrypoint()
def main(
    run_readme: str,
    config_name: str = "dp",
    task: str = "pusht",
    detach: bool = False,
    hydra_overrides: str = "",
) -> None:
    overrides = [part for part in hydra_overrides.split() if part.strip()] if hydra_overrides else []
    invoke_command = _build_invoke_command(
        config_name=config_name,
        task=task,
        detach=detach,
        hydra_overrides=hydra_overrides,
        run_readme=run_readme,
    )
    spawn_modal_function(
        train_dp,
        label=f"train_dp:{task}/{config_name}",
        wait=not detach,
        config_name=config_name,
        task=task,
        hydra_overrides=overrides or None,
        run_readme=run_readme,
        invoke_command=invoke_command,
    )
