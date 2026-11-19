import numpy as np

from tiago_door_planning.model.door_scene import DoorDimensions, scene_from_door_poses
from tiago_door_planning.model.geometry import make_pose
from tiago_door_planning.motion.compliance import (
    RADIAL,
    TANGENTIAL,
    VERTICAL,
    ComplianceParams,
    admittance_step,
    deadband,
    door_axes,
    offset_in_world,
)

PARAMS = ComplianceParams(3.0, 0.002, 0.04, 5.0)


def scene(push=True):
    dims = DoorDimensions(0.9, 0.04, 2.0, 0.15, 1.0)
    hinge = make_pose([1.0, 0.0, 0.0], 0.0)
    handle = make_pose([1.0, 0.8, 1.0], 0.0)
    return scene_from_door_poses(hinge, handle, "left", push, dims)


def test_deadband_removes_small_forces_and_keeps_the_sign():
    assert np.allclose(deadband([1.0, -2.9, 5.0, -7.0], 3.0), [0.0, 0.0, 2.0, -4.0])


def test_offset_moves_along_the_force_beyond_the_deadband():
    force = np.array([15.0, 0.0, -9.0])

    offset = admittance_step(np.zeros(3), force, 0.1, PARAMS)

    assert np.allclose(offset, [0.0024, 0.0, -0.0012])


def test_offset_never_moves_along_the_door_motion():
    offset = admittance_step(np.zeros(3), np.array([0.0, 30.0, 0.0]), 1.0, PARAMS)

    assert offset[TANGENTIAL] == 0.0


def test_offset_is_clipped_per_axis():
    offset = np.zeros(3)
    for _ in range(100):
        offset = admittance_step(offset, np.array([50.0, 0.0, -50.0]), 0.1, PARAMS)

    assert np.allclose(offset[[RADIAL, VERTICAL]], [0.04, -0.04])


def test_offset_leaks_back_without_force():
    offset = admittance_step(np.array([0.02, 0.0, -0.01]), np.zeros(3), 0.5, PARAMS)

    assert np.allclose(offset, [0.018, 0.0, -0.009])


def test_door_axes_follow_the_leaf_and_the_opening_direction():
    for push in (True, False):
        door = scene(push)
        axes = door_axes(door, 0.0)

        assert np.allclose(axes[RADIAL], [0.0, 1.0, 0.0])
        assert np.allclose(axes[VERTICAL], [0.0, 0.0, 1.0])
        turned = door_axes(door, 0.01)[RADIAL]
        assert (turned - axes[RADIAL]) @ axes[TANGENTIAL] > 0.0


def test_offset_in_world_combines_the_axes():
    axes = door_axes(scene(), 0.0)

    assert np.allclose(offset_in_world([0.01, 0.0, 0.02], axes), [0.0, 0.01, 0.02])
