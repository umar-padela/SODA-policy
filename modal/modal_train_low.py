"""
Local entrypoint — train π_low on Modal.

  modal run modal/modal_train_low.py --run-readme "A2 smoke: 25 epochs supervised pi_low"
  modal run modal/modal_train_low.py --config-name soda_supervised --task pusht --run-readme "..."
  modal run modal/modal_train_low.py --hydra-overrides "train_low.num_epochs=3" --run-readme "..."
  modal run modal/modal_train_low.py --detach --run-readme "..."

Remote: ``modal_config.train_low`` → ``python soda/training/train_low.py`` on GPU.
Canonical checkpoints: ``/experiments/train_low/{config_name}/`` on Volume ``soda-experiments``.
Each run is also archived under ``.../{config_name}/runs/{YYYYMMDD_HHMMSS}/`` with README.
"""

from modal_config import app, spawn_modal_function, train_low


def _build_invoke_command(
    *,
    config_name: str,
    task: str,
    detach: bool,
    hydra_overrides: str,
    run_readme: str,
) -> str:
    parts = [
        "modal run modal/modal_train_low.py",
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
    config_name: str = "soda_supervised",
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
        train_low,
        label=f"train_low:{task}/{config_name}",
        wait=not detach,
        config_name=config_name,
        task=task,
        hydra_overrides=overrides or None,
        run_readme=run_readme,
        invoke_command=invoke_command,
    )
