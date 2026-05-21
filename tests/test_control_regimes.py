"""Tests for control regime params (no sim / GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from soda.inference.control_regimes import (
    ControlParams,
    extract_execute_actions,
    resolve_control_params,
)


def test_default_action_horizon_is_one():
    c = resolve_control_params(regime="receding", n_obs_steps=2)
    assert c.action_horizon == 1
    assert c.action_start_index == 1


def test_action_horizon_override():
    c = resolve_control_params(
        regime="receding", n_obs_steps=2, action_horizon=8
    )
    assert c.action_horizon == 8


def test_extract_from_action_pred_dp_style():
    pred = np.arange(32, dtype=np.float32).reshape(1, 16, 2)
    control = ControlParams(action_horizon=8, action_start_index=1)
    out = extract_execute_actions({"action_pred": pred}, control)
    assert out.shape == (8, 2)
    np.testing.assert_array_equal(out[0], pred[0, 1])


def test_open_executes_full_chunk():
    pred = np.arange(32, dtype=np.float32).reshape(1, 16, 2)
    control = ControlParams(regime="open", action_horizon=1, action_start_index=1)
    out = extract_execute_actions({"action_pred": pred}, control)
    assert out.shape == (15, 2)


def test_action_horizon_cannot_exceed_available():
    pred = np.zeros((1, 4, 2), dtype=np.float32)
    control = ControlParams(action_horizon=8, action_start_index=0)
    with pytest.raises(ValueError):
        extract_execute_actions({"action_pred": pred}, control)


def test_invalid_action_horizon():
    with pytest.raises(ValueError):
        ControlParams(action_horizon=0)
