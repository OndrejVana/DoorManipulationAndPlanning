import numpy as np
import pytest

from tiago_door_planning.door_scene import (
    DoorDimensions,
    door_box,
    door_yaw,
    from_hinge_frame,
    handle_pose,
    handle_radius,
    opening_sign,
    scene_from_door_poses,
    to_hinge_frame,
    wall_boxes,
)
from tiago_door_planning.geometry import make_pose, point_to_box_distance

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)


def make_scene(push=False, hinge_side="left"):
    hinge = make_pose([1.0, 0.55, 0.0], 0.0)
    handle = make_pose([1.0, -0.25, 1.0], -np.pi / 2.0)
    return scene_from_door_poses(hinge, handle, hinge_side, push, DIMENSIONS)


@pytest.mark.parametrize("push, hinge_side, expected", [
    (True, "left", 1.0), (True, "right", -1.0), (False, "left", -1.0), (False, "right", 1.0),
])
def test_opening_sign_matches_thesis_convention(push, hinge_side, expected):
    assert opening_sign(push, hinge_side) == expected


def test_handle_radius_is_measured_distance():
    assert handle_radius(make_scene()) == pytest.approx(0.8)


def test_handle_pose_at_zero_is_detected_handle():
    scene = make_scene()

    assert handle_pose(scene, 0.0) == pytest.approx(scene.handle_closed)


def test_pull_door_swings_towards_robot():
    scene = make_scene(push=False)

    assert handle_pose(scene, np.pi / 2.0)[0, 3] == pytest.approx(1.0 - 0.8)


def test_open_door_is_perpendicular_to_wall():
    scene = make_scene()

    assert abs(np.cos(door_yaw(scene, np.pi / 2.0) - scene.closed_yaw)) < 1e-9


def test_open_door_clears_the_wall():
    scene = make_scene()
    tip = handle_pose(scene, np.pi / 2.0)[:3, 3]

    assert all(point_to_box_distance(tip, wall) > 0.0 for wall in wall_boxes(scene))


def test_closed_door_fills_the_doorway():
    scene = make_scene()
    doorway_center = np.array([1.02, 0.1, 1.0])

    assert point_to_box_distance(doorway_center, door_box(scene, 0.0)) < 0.0


def test_hinge_frame_round_trip():
    scene = make_scene()
    pose = np.array([0.3, -0.4, 1.2])

    assert from_hinge_frame(scene, to_hinge_frame(scene, pose)) == pytest.approx(pose)


def test_hinge_frame_y_points_into_swing_for_both_hinge_sides():
    for hinge_side in ("left", "right"):
        scene = make_scene(hinge_side=hinge_side)
        swung_handle = handle_pose(scene, 0.3)[:3, 3]
        local = to_hinge_frame(scene, [swung_handle[0], swung_handle[1], 0.0])

        assert local[1] > 0.0
