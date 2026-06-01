"""
Train SODA π_low + π_high on Block Push.

Usage:
  # π_low (run first)
  modal run --detach experiments/final_experiments/block_push/modal_train_soda.py \\
      --stage low --run-readme "block_push SODA pi_low k9"

  # π_high (after π_low finishes, supply checkpoint)
  modal run --detach experiments/final_experiments/block_push/modal_train_soda.py \\
      --stage high \\
      --low-checkpoint /experiments/train_low/pusht_block_push_soda_supervised/best.ckpt \\
      --run-readme "block_push SODA pi_high"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "modal"))
from modal_config import app, spawn_modal_function, train_low, train_high  # noqa: E402

CONFIG_NAME = "soda_supervised"
TASK        = "block_push"


@app.local_entrypoint()
def main(
    stage: str = "low",
    run_readme: str = "block_push SODA",
    low_checkpoint: str = None,
    hydra_overrides: str = "",
) -> None:
    assert stage in ("low", "high"), "--stage must be 'low' or 'high'"

    overrides = [p for p in hydra_overrides.split() if p.strip()] if hydra_overrides else []

    if stage == "low":
        print(f"Launching π_low training — task={TASK} config={CONFIG_NAME}")
        spawn_modal_function(
            train_low,
            label=f"train_low:{TASK}/{CONFIG_NAME}",
            wait=True,
            config_name=CONFIG_NAME,
            task=TASK,
            hydra_overrides=overrides or None,
            run_readme=run_readme,
            invoke_command=(
                f"modal run --detach experiments/final_experiments/block_push/modal_train_soda.py "
                f"--stage low --run-readme \"{run_readme}\""
            ),
        )
    else:
        if not low_checkpoint:
            raise ValueError("--low-checkpoint required for --stage high")
        print(f"Launching π_high training — low_checkpoint={low_checkpoint}")
        spawn_modal_function(
            train_high,
            label=f"train_high:{TASK}/{CONFIG_NAME}",
            wait=True,
            config_name=CONFIG_NAME,
            task=TASK,
            low_checkpoint=low_checkpoint,
            hydra_overrides=overrides or None,
            run_readme=run_readme,
            invoke_command=(
                f"modal run --detach experiments/final_experiments/block_push/modal_train_soda.py "
                f"--stage high --low-checkpoint {low_checkpoint} --run-readme \"{run_readme}\""
            ),
        )
