import numpy as np
import pytest

from tiago_door_planning.motion.trajectory import DoorTrajectory, from_angle, reference_at, split_by_sweep, straight_drive


def make_trajectory():
    base_poses = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.2]])
    return DoorTrajectory(base_poses, np.array([0.0, 0.1, 0.2]), 0.5, "right", np.zeros((3, 7)))


def test_reference_interpolates_between_samples():
    reference = reference_at(make_trajectory(), 0.5)

    assert reference.base_pose == pytest.approx([0.05, 0.0, 0.0])
    assert reference.door_angle == pytest.approx(0.05)


def test_reference_velocity_follows_sample_period():
    reference = reference_at(make_trajectory(), 1.5)

    assert reference.linear == pytest.approx(0.2)
    assert reference.angular == pytest.approx(0.4)


def test_reference_at_end_returns_last_sample():
    reference = reference_at(make_trajectory(), 2.0)

    assert reference.base_pose == pytest.approx([0.2, 0.0, 0.2])


def opening_trajectory(angles_deg):
    angles = np.radians(angles_deg)
    poses = np.column_stack([np.arange(len(angles)) * 0.1, np.zeros(len(angles)), np.zeros(len(angles))])
    return DoorTrajectory(poses, angles, 0.1, "right", np.zeros((len(angles), 7)))


def test_split_pieces_share_boundaries_and_respect_the_sweep():
    pieces = split_by_sweep(opening_trajectory(np.arange(0.0, 91.0, 5.0)), np.radians(45.0))

    assert [np.degrees(p.door_angles[[0, -1]]).round().tolist() for p in pieces] == [[0, 45], [45, 90]]
    assert np.allclose(pieces[0].base_poses[-1], pieces[1].base_poses[0])


def test_base_motion_without_door_motion_stays_in_one_piece():
    pieces = split_by_sweep(opening_trajectory([0.0, 10.0, 10.0, 10.0, 20.0]), np.radians(45.0))

    assert len(pieces) == 1
    assert len(pieces[0].door_angles) == 5


def test_samples_just_short_of_the_sweep_leave_no_tiny_last_piece():
    pieces = split_by_sweep(opening_trajectory(np.append(np.arange(0.0, 90.0, 0.7), 90.0)), np.radians(45.0))

    assert len(pieces) == 2
    assert np.degrees(pieces[0].door_angles[-1]) == pytest.approx(45.0, abs=0.5)


def test_from_angle_starts_at_the_last_sample_below_the_door():
    trajectory = DoorTrajectory(np.zeros((4, 3)), np.array([0.0, 0.1, 0.2, 0.3]), 0.1, "right", np.zeros((4, 7)))

    assert from_angle(trajectory, 0.15).door_angles == pytest.approx([0.1, 0.2, 0.3])
    assert from_angle(trajectory, 0.5).door_angles == pytest.approx([0.2, 0.3])
    assert from_angle(trajectory, -1.0).door_angles == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_reference_interpolates_the_planned_arm():
    trajectory = make_trajectory()._replace(arm_qs=np.array([np.zeros(7), np.ones(7), np.ones(7)]))

    assert reference_at(trajectory, 0.5).arm_q == pytest.approx(np.full(7, 0.5))


def test_straight_drive_moves_the_base_along_its_heading_only():
    start = np.array([1.0, 2.0, np.pi / 2.0])
    arm_q = np.arange(7.0)

    drive = straight_drive(start, -0.2, 0.5, arm_q, "left", 0.1)

    assert drive.base_poses[0] == pytest.approx(start)
    assert drive.base_poses[-1] == pytest.approx([1.0, 1.8, np.pi / 2.0])
    assert np.all(drive.door_angles == 0.5)
    assert drive.arm_qs.shape == (len(drive.door_angles), 7)


def test_from_angle_keeps_the_base_motion_before_the_door_moves():
    poses = np.column_stack([np.arange(5.0), np.zeros(5), np.zeros(5)])
    trajectory = DoorTrajectory(poses, np.array([0.5, 0.5, 0.5, 0.6, 0.7]), 0.1, "left", np.zeros((5, 7)))

    assert from_angle(trajectory, 0.52).base_poses[0][0] == 0.0
    assert from_angle(trajectory, 0.65).base_poses[0][0] == 3.0
