import numpy as np
import pytest

from tiago_door_planning.model.door_scene import (
    DoorDimensions,
    back_handle,
    back_scene,
    door_angle_at,
    door_box,
    door_yaw,
    handle_pose,
    opening_sign,
    scene_from_door_poses,
    swing_polygon,
    wall_boxes,
)
from tiago_door_planning.model.geometry import make_pose, points_to_box_distances

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)


def make_scene(push=False, hinge_side="left"):
    hinge = make_pose([1.0, 0.55, 0.0], 0.0)
    handle = make_pose([1.0, -0.25, 1.0], -np.pi / 2.0)
    return scene_from_door_poses(hinge, handle, hinge_side, push, DIMENSIONS)


@pytest.mark.parametrize(
    "push, hinge_side, expected",
    [(True, "left", 1.0), (True, "right", -1.0), (False, "left", -1.0), (False, "right", 1.0)],
)
def test_opening_sign_matches_thesis_convention(push, hinge_side, expected):
    assert opening_sign(push, hinge_side) == expected


def test_handle_pose_at_zero_is_detected_handle():
    scene = make_scene()

    assert handle_pose(scene, 0.0) == pytest.approx(scene.handle_closed)


@pytest.mark.parametrize("push", [True, False])
def test_door_angle_at_inverts_handle_pose(push):
    scene = make_scene(push=push)

    assert door_angle_at(scene, handle_pose(scene, 0.4)[:2, 3]) == pytest.approx(0.4)


def test_handle_standoff_does_not_tilt_the_door():
    handle = np.eye(4)
    handle[:3, :3] = [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    handle[:3, 3] = [0.66, 0.40, 1.05]
    scene = scene_from_door_poses(make_pose([0.75, -0.35, 0.0], 0.0), handle, "right", True, DIMENSIONS)

    assert door_yaw(scene, 0.0) == pytest.approx(np.pi / 2.0)


def test_pull_door_swings_towards_robot():
    scene = make_scene(push=False)

    assert handle_pose(scene, np.pi / 2.0)[0, 3] == pytest.approx(1.0 - 0.8)


def test_open_door_is_perpendicular_to_wall():
    scene = make_scene()

    assert abs(np.cos(door_yaw(scene, np.pi / 2.0) - scene.closed_yaw)) < 1e-9


def test_open_door_clears_the_wall():
    scene = make_scene()
    tip = handle_pose(scene, np.pi / 2.0)[:3, 3]

    assert all(points_to_box_distances([tip], wall)[0] > 0.0 for wall in wall_boxes(scene))


def test_closed_door_fills_the_doorway():
    scene = make_scene()
    doorway_center = np.array([1.02, 0.1, 1.0])

    assert points_to_box_distances([doorway_center], door_box(scene, 0.0))[0] < 0.0


def pull_handle_scene(y_up=False):
    """A pull door: handle z out of the face toward the robot (-x), y down, or up when a tag model turns it over."""
    hinge = make_pose([1.0, 0.55, 0.0], 0.0)
    handle = np.eye(4)
    handle[:3, :3] = np.column_stack([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    if y_up:
        handle[:3, :2] *= -1.0
    handle[:3, 3] = [0.9, -0.25, 1.0]
    return scene_from_door_poses(hinge, handle, "left", False, DIMENSIONS)


def test_back_handle_mirrors_through_the_leaf_and_faces_the_other_way():
    scene = pull_handle_scene()
    leaf = door_box(scene, 0.0)
    normal = leaf.pose[:3, 1]
    front = scene.handle_closed
    back = back_handle(scene)

    assert (back[:3, 3] - leaf.pose[:3, 3]) @ normal == pytest.approx(-(front[:3, 3] - leaf.pose[:3, 3]) @ normal)
    assert back[2, 3] == pytest.approx(front[2, 3])
    assert back[:3, 2] == pytest.approx(-front[:3, 2])
    assert back_handle(back_scene(scene)) == pytest.approx(front)


@pytest.mark.parametrize("y_up", [False, True])
def test_back_handle_is_upright_like_a_push_handle(y_up):
    back = back_handle(pull_handle_scene(y_up))

    assert back[:3, 1] == pytest.approx([0.0, 0.0, -1.0])
    assert back[:3, 2] == pytest.approx([1.0, 0.0, 0.0])
    assert np.linalg.det(back[:3, :3]) == pytest.approx(1.0)


def _inside(polygon, point):
    """Even-odd rule."""
    inside = False
    for (x1, y1), (x2, y2) in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (y1 > point[1]) != (y2 > point[1]):
            if point[0] < x1 + (point[1] - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


@pytest.mark.parametrize("push, hinge_side", [(True, "left"), (True, "right"), (False, "left"), (False, "right")])
def test_swing_polygon_covers_the_leaf_at_every_angle(push, hinge_side):
    scene = make_scene(push, hinge_side)
    polygon = swing_polygon(scene, np.radians(100.0), 0.1)

    for angle in np.radians([0.0, 30.0, 60.0, 90.0, 100.0]):
        box = door_box(scene, angle)
        for x in (-1.0, 1.0):
            for y in (-1.0, 1.0):
                corner = box.pose[:3, :3] @ (box.half_extents * [x, y, 0.0]) + box.pose[:3, 3]
                assert _inside(polygon, corner[:2])


def test_swing_polygon_leaves_the_other_side_of_the_wall_free():
    scene = make_scene(push=True)
    polygon = swing_polygon(scene, np.radians(100.0), 0.1)
    middle = handle_pose(scene, 0.0)[:2, 3] - 0.3 * (scene.handle_closed[:2, 3] - scene.hinge[:2])

    swing_side = handle_pose(scene, np.radians(60.0))[:2, 3]
    other_side = 2.0 * middle - swing_side

    assert _inside(polygon, swing_side)
    assert not _inside(polygon, other_side)
