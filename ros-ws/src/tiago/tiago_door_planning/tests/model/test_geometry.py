import numpy as np
import pytest

from tiago_door_planning.model.geometry import (
    Box,
    local_pose_error,
    make_pose,
    matrix_to_pose2d,
    points_to_box_distances,
    points_to_boxes_distances,
    pose2d_to_matrix,
    quaternion_to_rotation,
    rotation_to_quaternion,
    rotation_log,
    rotation_z,
    rpy_to_rotation,
)

UNIT_BOX = Box(np.eye(4), np.array([0.5, 0.5, 0.5]))


def test_pose2d_round_trip():
    pose2d = np.array([1.0, -2.0, 0.7])

    assert matrix_to_pose2d(pose2d_to_matrix(pose2d)) == pytest.approx(pose2d)


def test_point_outside_box_has_positive_distance():
    assert points_to_box_distances([[1.5, 0.0, 0.0]], UNIT_BOX)[0] == pytest.approx(1.0)


def test_point_inside_box_has_negative_distance():
    assert points_to_box_distances([[0.3, 0.0, 0.0]], UNIT_BOX)[0] == pytest.approx(-0.2)


def test_distance_respects_box_rotation():
    rotated = Box(make_pose([0.0, 0.0, 0.0], np.pi / 2.0), np.array([1.0, 0.1, 0.5]))

    assert points_to_box_distances([[0.0, 0.9, 0.0]], rotated)[0] < 0.0


def test_rotation_log_recovers_axis_angle():
    assert rotation_log(rotation_z(0.5)) == pytest.approx([0.0, 0.0, 0.5])


def test_local_pose_error_is_expressed_in_current_frame():
    current = make_pose([0.0, 0.0, 0.0], np.pi / 2.0)
    target = make_pose([0.0, 1.0, 0.0], np.pi / 2.0)

    assert local_pose_error(current, target) == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("rpy", [(0.0, 0.0, 0.0), (0.0, 0.0, np.pi), (np.pi, np.pi / 2.0, 0.0), (0.3, -1.2, 2.5)])
def test_quaternion_round_trip(rpy):
    rotation = rotation_z(rpy[2]) @ quaternion_to_rotation(np.sin(rpy[1] / 2.0), 0.0, 0.0, np.cos(rpy[1] / 2.0))
    rotation = rotation @ quaternion_to_rotation(0.0, np.sin(rpy[0] / 2.0), 0.0, np.cos(rpy[0] / 2.0))

    assert quaternion_to_rotation(*rotation_to_quaternion(rotation)) == pytest.approx(rotation)


def test_distances_to_several_boxes_match_one_at_a_time():
    boxes = [
        Box(make_pose([1.0, 0.0, 0.5], 0.3), np.array([0.2, 0.1, 0.5])),
        Box(make_pose([-1.0, 2.0, 1.0], -1.0), np.array([0.5, 0.05, 1.0])),
    ]
    points = np.array([[0.0, 0.0, 0.5], [1.0, 0.05, 0.4], [-1.0, 2.5, 3.0]])

    expected = np.column_stack([points_to_box_distances(points, box) for box in boxes])

    assert points_to_boxes_distances(points, boxes) == pytest.approx(expected)


def test_rpy_matches_fixed_axis_convention():
    roll, pitch, yaw = 0.3, -0.4, 1.1
    rx, ry, rz = rpy_to_rotation(roll, 0.0, 0.0), rpy_to_rotation(0.0, pitch, 0.0), rpy_to_rotation(0.0, 0.0, yaw)

    assert np.allclose(rpy_to_rotation(roll, pitch, yaw), rz @ ry @ rx)
    assert np.allclose(rz @ [1.0, 0.0, 0.0], [np.cos(yaw), np.sin(yaw), 0.0])
