from pathlib import Path
import importlib
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
READER_DIR = REPO_ROOT / "oculus_reader"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(READER_DIR))

coordinate_conversion = importlib.import_module(
    "oculus_reader.coordinate_conversion"
)
QUEST_TO_ROS_BASIS = coordinate_conversion.QUEST_TO_ROS_BASIS
quest_pose_to_ros = coordinate_conversion.quest_pose_to_ros


def pose(rotation=None, translation=(0.0, 0.0, 0.0)):
    transform = np.eye(4)
    if rotation is not None:
        transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def test_basis_is_a_proper_rotation():
    np.testing.assert_allclose(
        QUEST_TO_ROS_BASIS.T @ QUEST_TO_ROS_BASIS, np.eye(3)
    )
    assert np.isclose(np.linalg.det(QUEST_TO_ROS_BASIS), 1.0)


def test_translation_maps_right_up_forward_to_ros_left_up_forward():
    translation, _ = quest_pose_to_ros(pose(translation=(1.0, 2.0, -3.0)))
    np.testing.assert_allclose(translation, (3.0, -1.0, 2.0))


@pytest.mark.parametrize("quest_axis", np.eye(3))
def test_rotation_axis_uses_the_same_basis_as_translation(quest_axis):
    angle = 0.37
    quest_rotation = Rotation.from_rotvec(angle * quest_axis).as_matrix()
    _, ros_rotation = quest_pose_to_ros(pose(rotation=quest_rotation))

    expected_axis = QUEST_TO_ROS_BASIS @ quest_axis
    expected_rotation = Rotation.from_rotvec(angle * expected_axis).as_matrix()
    np.testing.assert_allclose(ros_rotation, expected_rotation, atol=1e-12)


def test_identity_pose_has_identity_axes_after_change_of_basis():
    _, rotation = quest_pose_to_ros(np.eye(4))
    np.testing.assert_allclose(rotation, np.eye(3))


def test_upside_down_mount_rotates_complete_pose_about_ros_forward_axis():
    mount = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    translation, rotation = quest_pose_to_ros(
        pose(translation=(1.0, 2.0, -3.0)), mount
    )

    np.testing.assert_allclose(translation, (3.0, 1.0, -2.0), atol=1e-12)
    np.testing.assert_allclose(rotation, mount, atol=1e-12)


def test_mount_calibration_maps_relative_rotation_with_translation_basis():
    mount = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    initial = Rotation.from_euler("xyz", (0.2, -0.1, 0.3)).as_matrix()
    current = Rotation.from_euler("xyz", (-0.3, 0.4, 0.1)).as_matrix()
    _, initial_ros = quest_pose_to_ros(pose(rotation=initial), mount)
    _, current_ros = quest_pose_to_ros(pose(rotation=current), mount)

    source_to_parent = mount @ QUEST_TO_ROS_BASIS
    expected_delta = (
        source_to_parent @ current @ initial.T @ source_to_parent.T
    )
    np.testing.assert_allclose(
        current_ros @ initial_ros.T, expected_delta, atol=1e-12
    )


def test_invalid_rotation_is_rejected():
    transform = np.eye(4)
    transform[0, 0] = 2.0
    with pytest.raises(ValueError, match="proper rotation"):
        quest_pose_to_ros(transform)


def test_invalid_mount_rotation_is_rejected():
    with pytest.raises(ValueError, match="Headset mount rotation"):
        quest_pose_to_ros(np.eye(4), np.diag((1.0, 1.0, 2.0)))
