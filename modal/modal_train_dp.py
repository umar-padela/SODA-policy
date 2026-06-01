"""
Train vanilla Diffusion Policy on Modal using Columbia's pipeline directly.

Uses Columbia's train.py + Hydra config system — no custom config translation.
The only SODA addition is a gym 0.23 compatibility patch (shared_memory=False).

Predefined configs:
  dp     — Columbia baseline: kernel_size=5, 3050 epochs
  dp_k9  — kernel_size=9, 500 epochs

Usage:
  modal run --detach modal/modal_train_dp.py --config-name dp_k9 --run-readme "DP k=9 baseline"
  modal run --detach modal/modal_train_dp.py --config-name dp    --run-readme "DP k=5 Columbia baseline"

Custom overrides (appended on top of the named config):
  modal run --detach modal/modal_train_dp.py --config-name dp_k9 --run-readme "..." \\
    --extra-overrides "training.num_epochs=200 policy.kernel_size=7"
"""

from modal_config import app, spawn_modal_function, train_dp, EXPERIMENTS_MOUNT

# Overrides shared by all PushT DP runs — matches Columbia's released checkpoint config
_BASE = [
    "--config-name=train_diffusion_unet_hybrid_workspace",
    "task=pusht_image",
    "task.dataset.zarr_path=data/raw/pusht/pusht.zarr",
    "policy.crop_shape=[84,84]",
    "logging.project=soda-train-dp",
    "logging.mode=online",
    "training.resume=False",
]

_CONFIGS = {
    "dp": [
        "policy.kernel_size=5",
        "training.num_epochs=3050",
        f"hydra.run.dir={EXPERIMENTS_MOUNT}/final_experiments/pusht/comparison_study/dp",
        "logging.name=dp_pusht_image",
    ],
    "dp_k9": [
        "policy.kernel_size=9",
        "training.num_epochs=500",
        f"hydra.run.dir={EXPERIMENTS_MOUNT}/final_experiments/pusht/comparison_study/dp_k9",
        "logging.name=dp_k9_pusht_image",
    ],
}


@app.local_entrypoint()
def main(
    run_readme: str,
    config_name: str = "dp_k9",
    extra_overrides: str = "",
    detach: bool = False,
) -> None:
    if config_name not in _CONFIGS:
        raise ValueError(f"Unknown config '{config_name}'. Options: {list(_CONFIGS)}")

    overrides = _BASE + _CONFIGS[config_name]
    if extra_overrides.strip():
        overrides += extra_overrides.strip().split()

    invoke_command = (
        f"modal run {'--detach ' if detach else ''}modal/modal_train_dp.py "
        f"--config-name {config_name} --run-readme \"{run_readme}\""
    )

    spawn_modal_function(
        train_dp,
        label=f"train_dp:{config_name}",
        wait=not detach,
        hydra_overrides=overrides,
        run_readme=run_readme,
        invoke_command=invoke_command,
    )
