"""Camera transforms shared by CPU-only data preparation paths."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def look_at_to_c2w(
    camera_position: Sequence[float],
    target_position: Sequence[float] = (0.0, 0.0, 0.0),
    up_dir: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return the legacy RF1 camera-to-world look-at transform."""
    camera_direction = np.asarray(camera_position) - np.asarray(target_position)
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    camera_right = np.cross(np.asarray(up_dir), camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)

    rotation_transform = np.zeros((4, 4))
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[-1, -1] = 1.0

    translation_transform = np.eye(4)
    translation_transform[:3, -1] = -np.asarray(camera_position)
    look_at_transform = rotation_transform @ translation_transform
    return np.linalg.inv(look_at_transform)
