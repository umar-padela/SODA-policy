"""
Local entrypoint — train π_low on Modal.

  modal run modal/modal_train_low.py --run-readme "..."          # live logs; job dies if terminal closes
  modal run --detach modal/modal_train_low.py --run-readme "..." # live logs; job survives if terminal closes

The ``--detach`` flag MUST come before the script path to be interpreted as a
Modal CLI flag.  Placed after the script it becomes a function argument instead
and has no effect on job survival — a common mistake.

``--detach`` behaviour:
  without: Modal stops the ephemeral app when this entrypoint exits → job killed.
  with:    Modal keeps the spawned job running after the local client disconnects.
           The entrypoint still blocks (showing live logs) until you close the
           terminal or Ctrl-C; the remote job then continues independently.

Remote: ``modal_config.train_low`` → ``python soda/training/train_low.py`` on GPU.
Canonical checkpoints: ``/experiments/train_low/{config_name}/`` on Volume ``soda-experiments``.
Each run is also archived under ``.../{config_name}/runs/{YYYYMMDD_HHMMSS}/`` with README.
"""

from modal_config import app, spawn_modal_function, train_low


def _build_invoke_command(
    *,
    config_name: str,
    task: str,
    hydra_overrides: str,
    run_readme: str,
) -> str:
    parts = [
        "modal run --detach modal/modal_train_low.py",
        f"--config-name {config_name}",
        f"--task {task}",
    ]
    if hydra_overrides.strip():
        parts.append(f'--hydra-overrides "{hydra_overrides.strip()}"')
    parts.append(f'--run-readme "{run_readme}"')
    return " ".join(parts)


@app.local_entrypoint()
def main(
    run_readme: str,
    config_name: str = "soda_supervised",
    task: str = "pusht",
    hydra_overrides: str = "",
) -> None:
    overrides = [part for part in hydra_overrides.split() if part.strip()] if hydra_overrides else []
    invoke_command = _build_invoke_command(
        config_name=config_name,
        task=task,
        hydra_overrides=hydra_overrides,
        run_readme=run_readme,
    )
    spawn_modal_function(
        train_low,
        label=f"train_low:{task}/{config_name}",
        wait=True,
        config_name=config_name,
        task=task,
        hydra_overrides=overrides or None,
        run_readme=run_readme,
        invoke_command=invoke_command,
    )
