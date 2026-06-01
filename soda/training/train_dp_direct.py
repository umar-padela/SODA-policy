"""
Train vanilla Diffusion Policy using Columbia's own pipeline unchanged.

The only thing this file does beyond Columbia's train.py:
  - Patches AsyncVectorEnv default shared_memory=False for gym 0.23+ compatibility

All sys.argv arguments pass through to Columbia's Hydra app unchanged.

Example:
  python soda/training/train_dp_direct.py \\
    --config-name=train_diffusion_unet_hybrid_workspace \\
    task=pusht_image \\
    task.dataset.zarr_path=data/raw/pusht/pusht.zarr \\
    policy.kernel_size=9 \\
    training.num_epochs=500 \\
    'hydra.run.dir=/experiments/final_experiments/pusht/comparison_study/dp_k9'
"""
import os
import runpy
from pathlib import Path


def _patch_async_vector_env() -> None:
    """
    Patch Columbia's AsyncVectorEnv for gym 0.23+ compatibility.

    gym 0.21 (Columbia's version) vs gym 0.23 (Modal image) has two API breaks:
      1. shared_memory=True default fails for PushT's OrderedDict obs space
      2. VectorEnv.reset() now calls reset_async(seed=, return_info=, options=)
         and reset_wait(timeout=, seed=, return_info=, options=) — Columbia's
         signatures take neither.
    """
    from diffusion_policy.gym_util import async_vector_env as _ave

    # Fix 1: shared_memory default
    _orig_init = _ave.AsyncVectorEnv.__init__

    def _patched_init(self, env_fns, dummy_env_fn=None, observation_space=None,
                      action_space=None, shared_memory=False, copy=True,
                      context=None, daemon=True, worker=None):
        _orig_init(self, env_fns, dummy_env_fn=dummy_env_fn,
                   observation_space=observation_space, action_space=action_space,
                   shared_memory=shared_memory, copy=copy,
                   context=context, daemon=daemon, worker=worker)

    _ave.AsyncVectorEnv.__init__ = _patched_init

    # Fix 2: reset_async / reset_wait new-signature kwargs
    _orig_reset_async = _ave.AsyncVectorEnv.reset_async

    def _patched_reset_async(self, seed=None, return_info=False, options=None):
        return _orig_reset_async(self)

    _ave.AsyncVectorEnv.reset_async = _patched_reset_async

    _orig_reset_wait = _ave.AsyncVectorEnv.reset_wait

    def _patched_reset_wait(self, timeout=None, seed=None, return_info=False, options=None):
        return _orig_reset_wait(self, timeout=timeout)

    _ave.AsyncVectorEnv.reset_wait = _patched_reset_wait


_patch_async_vector_env()

_TRAIN_PY = Path(__file__).parents[2] / "third_party" / "diffusion_policy" / "train.py"
runpy.run_path(str(_TRAIN_PY), run_name="__main__")
os._exit(0)
