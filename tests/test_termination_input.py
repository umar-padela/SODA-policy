"""Tests for ``low_policy.termination_input`` (obs vs bottleneck β features)."""

from __future__ import annotations

import inspect

import pytest

from soda.models.low_policy import (
    normalize_termination_input,
    obs_termination_input_dim,
)


def test_normalize_termination_input():
    assert normalize_termination_input("bottleneck") == "bottleneck"
    assert normalize_termination_input("OBS") == "obs"
    with pytest.raises(ValueError, match="termination_input"):
        normalize_termination_input("plan")


def test_obs_termination_input_dim():
    assert obs_termination_input_dim(64, 2, 32) == 160


def test_low_policy_supports_termination_input_flag():
    from soda.models import low_policy as lp_mod

    source = inspect.getsource(lp_mod._build_low_policy_class)
    assert "termination_input" in source
    assert "termination_stop_grad" in source
    assert "termination_features_from_batch" in source
    assert 'self.termination_input == "obs"' in source
    assert "self.cfg.termination_stop_grad" in source


def test_compute_loss_uses_termination_features_helper():
    from soda.models import low_policy as lp_mod

    source = inspect.getsource(lp_mod._build_low_policy_class)
    assert "termination_features_from_batch" in source
    assert "predict_termination_logit(term_features" in source
