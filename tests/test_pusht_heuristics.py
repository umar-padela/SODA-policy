"""Tests for Push-T supervised skill heuristics."""

from __future__ import annotations

import numpy as np
import pytest

from soda.option_discovery.supervised.pusht.heuristics import (
    SKILL_NAMES,
    label_skills,
    validate_skill_labels,
)


def test_skill_ids_are_contiguous_three_skills():
    assert list(SKILL_NAMES.keys()) == [0, 1, 2]


def test_validate_skill_labels_rejects_gaps():
    with pytest.raises(ValueError, match="Expected skill ids"):
        validate_skill_labels(np.array([0, 1, 3], dtype=np.int32))


def test_idle_episode_is_all_reposition():
    fake = {
        "state": np.zeros((20, 5), dtype=np.float32),
        "episode_ends": np.array([20], dtype=np.int32),
    }
    labels = label_skills(fake)
    assert np.all(labels == 0)
