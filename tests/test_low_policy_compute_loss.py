"""Tests for π_low ``compute_loss`` termination forward protocol."""

from __future__ import annotations

import inspect


def test_compute_loss_termination_uses_feature_helper_not_full_no_grad_on_mlp():
    """β MLP must sit outside feature ``no_grad`` (see ``termination_features_from_batch``)."""
    from soda.models import low_policy as lp_mod

    source = inspect.getsource(lp_mod._build_low_policy_class)
    assert "termination_features_from_batch" in source
    assert "predict_termination_logit(term_features)" in source
    assert "forward_unet_with_bottleneck" in source
