import numpy as np
import pytest

from tiago_door_planning.door_scene import DoorDimensions, scene_from_door_poses
from tiago_door_planning.geometry import make_pose
from tiago_door_planning.trajectory import (
    DoorTrajectory,
    load_trajectory,
    reference_at,
    save_trajectory,
    trajectory_to_world,
)

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)


def make_trajectory():
    base_poses = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.2]])
    return DoorTrajectory(base_poses, np.array([0.0, 0.1, 0.2]), 0.5, "right", np.zeros(7))


def make_scene(hinge_side):
    hinge = make_pose([1.0, 0.55, 0.0], 0.0)
    handle = make_pose([1.0, -0.25, 1.0], 0.0)
    return scene_from_door_poses(hinge, handle, hinge_side, False, DIMENSIONS)


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


def test_arm_swaps_for_door_opening_the_other_way():
    sides = {trajectory_to_world(make_trajectory(), make_scene(h)).side for h in ("left", "right")}

    assert sides == {"left", "right"}


def test_save_and_load_round_trip(tmpdir):
    path = str(tmpdir.join("trajectory.npz"))

    save_trajectory(path, make_trajectory())
    loaded = load_trajectory(path)

    assert loaded.base_poses == pytest.approx(make_trajectory().base_poses)
    assert loaded.side == "right"
