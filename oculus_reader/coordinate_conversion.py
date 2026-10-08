"""Coordinate conversion for Quest poses published to ROS.

Quest tracking uses +X right, +Y up, and -Z forward.  The ROS bridge uses
+X forward, +Y left, and +Z up.  A pose is a change of basis, so the same
matrix must convert both its translation and its rotation.
"""

import numpy as np


QUEST_TO_ROS_BASIS = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)


def _validated_rotation(rotation, label):
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-2)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=5e-2)
    ):
        raise ValueError(f"{label} must be a proper rotation")
    return rotation


def quest_pose_to_ros(transform, headset_mount_rotation_ros=None):
    """Return ROS translation and rotation for a Quest 4x4 pose matrix.

    ``headset_mount_rotation_ros`` calibrates the physical headset mounting in
    the ROS parent frame.  It left-multiplies the complete pose; the controller
    child-frame convention remains the basis-converted Quest convention.
    """
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Quest pose must be a finite 4x4 matrix")

    rotation = _validated_rotation(
        transform[:3, :3], "Quest pose rotation"
    )
    if headset_mount_rotation_ros is None:
        headset_mount_rotation_ros = np.eye(3)
    mount_rotation = _validated_rotation(
        headset_mount_rotation_ros, "Headset mount rotation"
    )

    source_to_parent = mount_rotation @ QUEST_TO_ROS_BASIS
    translation_ros = source_to_parent @ transform[:3, 3]
    rotation_ros = (
        source_to_parent @ rotation @ QUEST_TO_ROS_BASIS.T
    )
    return translation_ros, rotation_ros
