"""
Local entrypoint — train π_high on Modal.

  modal run modal/modal_train_high.py --run-readme "B5 full pi_high supervised" \\
    --low-checkpoint /experiments/train_low/soda_supervised/best.ckpt

π_high requires a trained π_low checkpoint (``--low-checkpoint`` or
``train_high.low_checkpoint=...`` in ``--hydra-overrides``). Checkpoints land at
``/experiments/train_high/{config_name}/`` on Volume ``soda-experiments``.
Archived runs: ``.../{config_name}/runs/{YYYYMMDD_HHMMSS}/``.
"""

from modal_config import app, spawn_modal_function, train_high


def _build_invoke_command(
    *,
    config_name: str,
    task: str,
    low_checkpoint: str | None,
    detach: bool,
    hydra_overrides: str,
    run_readme: str,
) -> str:
    parts = [
        "modal run modal/modal_train_high.py",
        f"--config-name {config_name}",
        f"--task {task}",
    ]
    if low_checkpoint:
        parts.append(f"--low-checkpoint {low_checkpoint}")
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
    low_checkpoint: str | None = None,
    detach: bool = False,
    hydra_overrides: str = "",
) -> None:
    overrides = [part for part in hydra_overrides.split() if part.strip()] if hydra_overrides else []
    invoke_command = _build_invoke_command(
        config_name=config_name,
        task=task,
        low_checkpoint=low_checkpoint,
        detach=detach,
        hydra_overrides=hydra_overrides,
        run_readme=run_readme,
    )
    spawn_modal_function(
        train_high,
        label=f"train_high:{task}/{config_name}",
        wait=not detach,
        config_name=config_name,
        task=task,
        low_checkpoint=low_checkpoint,
        hydra_overrides=overrides or None,
        run_readme=run_readme,
        invoke_command=invoke_command,
    )
