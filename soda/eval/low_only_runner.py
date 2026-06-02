"""
Push-T eval for trained π_low with a fixed skill id ω (no π_high / β).

Used in training_plan.md A3: qualitative rollouts before hierarchical stack is ready.
Executes duration-decoded **native-time** actions. With ``n_action_steps>1``,
``LowPolicy.predict_action`` returns the capped native chunk directly; with
``n_action_steps=1``, ``NativeChunkRolloutPolicy`` walks the full native chunk
before re-diffusing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soda.eval.dp_runner import build_pusht_image_runner
from soda.eval.metrics import DEFAULT_OVERLAP_CHECKPOINTS
from soda.eval.policy_loaders import load_low_policy_and_cfg
from soda.eval.runner_common import resolve_test_start_seed, serialize_runner_log
from soda.eval.run_naming import DEFAULT_TEST_START_SEED


class FixedOptionLowPolicy:
    """
    Adapter so Columbia ``PushTImageRunner`` can drive π_low.

    The runner calls ``predict_action(obs_dict)``; π_low requires ``option_id``.
    """

    def __init__(self, low_policy: Any, fixed_option_id: int) -> None:
        self._policy = low_policy
        self._fixed_option_id = int(fixed_option_id)

    @property
    def device(self) -> Any:
        return self._policy.device

    @property
    def dtype(self) -> Any:
        return self._policy.dtype

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        import torch

        value = next(iter(obs_dict.values()))
        batch_size = int(value.shape[0])
        option_id = torch.full(
            (batch_size,),
            self._fixed_option_id,
            device=self._policy.device,
            dtype=torch.long,
        )
        return self._policy.predict_action(obs_dict, option_id)


class NativeChunkRolloutPolicy:
    """
    Closed-loop cache for ``n_action_steps=1`` rollouts.

    π_low returns only the first native row per diffusion sample; this wrapper
    walks the full ``action_unstretched`` chunk before re-diffusing.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cached_native: Any | None = None
        self._cursor = 0
        self._last_out: dict[str, Any] = {}

    @property
    def device(self) -> Any:
        return self._inner.device

    @property
    def dtype(self) -> Any:
        return self._inner.dtype

    def reset(self) -> None:
        self._cached_native = None
        self._cursor = 0
        self._last_out = {}
        reset = getattr(self._inner, "reset", None)
        if callable(reset):
            reset()

    def predict_action(self, obs_dict: dict[str, Any]) -> dict[str, Any]:
        if self._cached_native is None or self._cursor >= int(
            self._cached_native.shape[1]
        ):
            out = self._inner.predict_action(obs_dict)
            native = out.get("action_unstretched")
            if native is None:
                self._cached_native = None
                self._cursor = 0
                return out

            self._cached_native = native
            self._cursor = 0
            self._last_out = out

        assert self._cached_native is not None
        end = self._cursor + 1
        chunk = self._cached_native[:, self._cursor:end]
        self._cursor = end

        return {
            **self._last_out,
            "action": chunk,
            "action_unstretched": self._cached_native,
        }


def wrap_low_policy_for_rollout(
    low_policy: Any,
    fixed_option_id: int,
    *,
    n_action_steps: int = 8,
    env_action_dim: int = 2,
) -> Any:
    """Fixed ω adapter; caches native chunks only when ``n_action_steps=1``."""
    _ = env_action_dim
    wrapped = FixedOptionLowPolicy(low_policy, fixed_option_id)
    if int(n_action_steps) == 1:
        return NativeChunkRolloutPolicy(wrapped)
    return wrapped


def run_low_only_pusht_eval(
    *,
    checkpoint: Path,
    fixed_option_id: int,
    output_dir: Path,
    device: str = "cuda:0",
    config_name: str = "soda_supervised",
    n_test: int = 5,
    n_train: int = 0,
    n_test_vis: int | None = None,
    n_train_vis: int = 0,
    n_action_steps: int = 1,
    max_steps: int = 300,
    test_start_seed: int = DEFAULT_TEST_START_SEED,
    train_start_seed: int = 0,
    record_video: bool = True,
    overlap_checkpoints: tuple[int, ...] = DEFAULT_OVERLAP_CHECKPOINTS,
    noise_eta: float = 0.0,
    noise_rho: float = 0.9,
    noise_scale_by_magnitude: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load π_low, hold ``fixed_option_id`` constant for the episode, run sim eval.

    Returns ``(soda_metrics, runner_log)`` — same contract as ``run_dp_pusht_eval``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "media").mkdir(parents=True, exist_ok=True)

    policy, _settings, cfg = load_low_policy_and_cfg(
        checkpoint,
        device=device,
        config_name=config_name,
    )
    seed = resolve_test_start_seed(test_start_seed, cfg)

    if n_test_vis is None:
        n_test_vis = min(n_test, 4) if record_video else 0
    if not record_video:
        n_test_vis = 0

    wrapped = wrap_low_policy_for_rollout(
        policy,
        fixed_option_id,
        n_action_steps=int(n_action_steps),
    )
    runner = build_pusht_image_runner(
        cfg,
        output_dir,
        n_test=n_test,
        n_train=n_train,
        n_test_vis=n_test_vis,
        n_train_vis=n_train_vis,
        n_action_steps=int(n_action_steps),
        max_steps=max_steps,
        test_start_seed=seed,
        train_start_seed=train_start_seed,
        overlap_checkpoints=overlap_checkpoints,
        noise_eta=noise_eta,
        noise_rho=noise_rho,
        noise_scale_by_magnitude=noise_scale_by_magnitude,
    )

    runner_log = runner.run(wrapped)
    soda_metrics = runner_log.pop("soda_metrics")
    soda_metrics["fixed_option_id"] = int(fixed_option_id)
    return soda_metrics, serialize_runner_log(runner_log)
