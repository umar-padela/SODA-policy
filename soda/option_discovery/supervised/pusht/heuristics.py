"""Heuristic option labels for Push-T (state-based)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Skill ids: 0 REPOSITION, 1 LINEAR-PUSH, 2 PIVOT-PUSH
GRIPPER_MOVE_THRESHOLD = 0.01
BLOCK_MOVE_THRESHOLD = 0.2
BLOCK_ROT_THRESHOLD = 0.1
PIVOT_AVG_THRESHOLD = 1.0

SKILL_NAMES = {
    0: "REPOSITION",
    1: "LINEAR-PUSH",
    2: "PIVOT-PUSH",
}


def validate_skill_labels(labels: np.ndarray) -> None:
    """Ensure labels use every skill id in ``SKILL_NAMES`` contiguously from 0."""
    labels = np.asarray(labels, dtype=np.int32)
    unique = sorted(int(u) for u in np.unique(labels))
    expected = sorted(SKILL_NAMES.keys())
    if unique != expected:
        raise ValueError(
            f"Expected skill ids {expected} ({[SKILL_NAMES[i] for i in expected]}), "
            f"got {unique}"
        )


def label_skills(raw_data: Mapping[str, Any]) -> np.ndarray:
    """
    Label each frame with a discrete skill id from state kinematics.

    Expects ``raw_data['state']`` with block rotation in **degrees** (index 4)
    and ``raw_data['episode_ends']`` cumulative episode end indices.
    """
    states = raw_data["state"]
    episode_ends = raw_data["episode_ends"]
    num_frames = states.shape[0]
    labels = np.zeros(num_frames, dtype=np.int32)

    start_idx = 0
    for ep_end in episode_ends:
        ep_states = states[start_idx:ep_end]
        n_ep_frames = len(ep_states)
        ep_labels = np.zeros(n_ep_frames, dtype=np.int32)

        g_vels = np.linalg.norm(ep_states[1:, 0:2] - ep_states[:-1, 0:2], axis=1)
        b_vels = np.linalg.norm(ep_states[1:, 2:4] - ep_states[:-1, 2:4], axis=1)
        raw_rot_diffs = np.abs(ep_states[1:, 4] - ep_states[:-1, 4])
        b_rot_vels = np.minimum(raw_rot_diffs, 360 - raw_rot_diffs)

        for i in range(n_ep_frames - 1):
            g_moving = g_vels[i] > GRIPPER_MOVE_THRESHOLD
            b_moving = (b_vels[i] > BLOCK_MOVE_THRESHOLD) or (
                b_rot_vels[i] > BLOCK_ROT_THRESHOLD
            )
            if g_moving and b_moving:
                ep_labels[i] = -1
            else:
                ep_labels[i] = 0

        i = 0
        while i < n_ep_frames:
            if ep_labels[i] == -1:
                seg_start = i
                while i < n_ep_frames and ep_labels[i] == -1:
                    i += 1
                seg_end = i
                if seg_end > seg_start + 1:
                    avg_rot = np.mean(b_rot_vels[seg_start : seg_end - 1])
                    skill = 2 if avg_rot >= PIVOT_AVG_THRESHOLD else 1
                else:
                    skill = 1
                ep_labels[seg_start:seg_end] = skill
            else:
                i += 1

        if n_ep_frames > 1:
            ep_labels[-1] = ep_labels[-2]
        labels[start_idx:ep_end] = ep_labels
        start_idx = ep_end

    return labels
