import numpy as np
import pytest

from tiago_door_planning.geometry import (
    Box,
    damped_pseudo_inverse,
    invert,
    local_pose_error,
    make_pose,
    matrix_to_pose2d,
    point_to_box_distance,
    pose2d_to_matrix,
    quaternion_to_rotation,
    rotation_log,
    rotation_z,
    sphere_to_box_distance,
    wrap_angle,
)

UNIT_BOX = Box(np.eye(4), np.array([0.5, 0.5, 0.5]))


def test_wrap_angle_maps_into_minus_pi_pi():
    assert wrap_angle(3.0 * np.pi / 2.0) == pytest.approx(-np.pi / 2.0)


def test_pose2d_round_trip():
    pose2d = np.array([1.0, -2.0, 0.7])

    assert matrix_to_pose2d(pose2d_to_matrix(pose2d)) == pytest.approx(pose2d)


def test_invert_gives_identity():
    pose = make_pose([1.0, 2.0, 3.0], 0.4)

    assert pose @ invert(pose) == pytest.approx(np.eye(4))


def test_point_outside_box_has_positive_distance():
    assert point_to_box_distance([1.5, 0.0, 0.0], UNIT_BOX) == pytest.approx(1.0)


def test_point_inside_box_has_negative_distance():
    assert point_to_box_distance([0.3, 0.0, 0.0], UNIT_BOX) == pytest.approx(-0.2)


def test_distance_respects_box_rotation():
    rotated = Box(make_pose([0.0, 0.0, 0.0], np.pi / 2.0), np.array([1.0, 0.1, 0.5]))

    assert point_to_box_distance([0.0, 0.9, 0.0], rotated) < 0.0


def test_sphere_distance_subtracts_radius():
    assert sphere_to_box_distance([1.5, 0.0, 0.0], 0.25, UNIT_BOX) == pytest.approx(0.75)


def test_yaw_quaternion_gives_z_rotation():
    half = 0.3
    rotation = quaternion_to_rotation(0.0, 0.0, np.sin(half), np.cos(half))

    assert rotation == pytest.approx(rotation_z(2.0 * half))


def test_rotation_log_recovers_axis_angle():
    assert rotation_log(rotation_z(0.5)) == pytest.approx([0.0, 0.0, 0.5])


def test_local_pose_error_is_expressed_in_current_frame():
    current = make_pose([0.0, 0.0, 0.0], np.pi / 2.0)
    target = make_pose([0.0, 1.0, 0.0], np.pi / 2.0)

    assert local_pose_error(current, target) == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_damped_pseudo_inverse_approaches_inverse():
    jacobian = np.array([[2.0, 0.0], [0.0, 4.0]])

    assert damped_pseudo_inverse(jacobian, 1e-6) == pytest.approx(np.linalg.inv(jacobian))
