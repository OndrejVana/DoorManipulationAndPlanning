import numpy as np
import pytest

from tiago_door_planning.model.collision import (
    FREE,
    GRASP,
    CollisionParams,
    arms_collide,
    base_costmap_clearance,
    distance_grid,
    robot_collides,
    spheres_in_world,
)
from tiago_door_planning.model.door_scene import DoorDimensions, door_box, handle_pose, scene_from_door_poses, wall_boxes
from tiago_door_planning.model.geometry import Sphere, make_pose

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)
PARAMS = CollisionParams(base_radius=0.3, base_center_height=0.15, margin=0.05)
FAR_BASE = np.array([-0.5, 0.0, 0.0])


def make_scene():
    hinge = make_pose([1.0, 0.55, 0.0], 0.0)
    handle = make_pose([1.0, -0.25, 1.0], -np.pi / 2.0)
    return scene_from_door_poses(hinge, handle, "left", False, DIMENSIONS)


def gripper_on_handle(scene):
    return [Sphere(handle_pose(scene, 0.0)[:3, 3], 0.05, True)]


def test_gripper_may_touch_door_while_grasping():
    scene = make_scene()

    assert not robot_collides([], door_box(scene, 0.0), None, FAR_BASE, gripper_on_handle(scene), GRASP, PARAMS)


def test_gripper_touching_door_collides_when_free():
    scene = make_scene()

    assert robot_collides([], door_box(scene, 0.0), None, FAR_BASE, gripper_on_handle(scene), FREE, PARAMS)


def test_elbow_inside_wall_collides_even_while_grasping():
    scene = make_scene()
    elbow = [Sphere(np.array([1.05, 0.9, 1.0]), 0.07, False)]

    assert robot_collides(wall_boxes(scene), door_box(scene, 0.0), None, FAR_BASE, elbow, GRASP, PARAMS)


def test_robot_far_from_door_is_free():
    scene = make_scene()
    elbow = [Sphere(np.array([-0.3, 0.0, 1.0]), 0.07, False)]

    assert not robot_collides(wall_boxes(scene), door_box(scene, 0.0), None, FAR_BASE, elbow, GRASP, PARAMS)


def test_spheres_follow_base_pose():
    spheres = spheres_in_world(np.array([1.0, 0.0, np.pi / 2.0]), [Sphere(np.array([0.5, 0.0, 1.0]), 0.1, False)])

    assert spheres[0].center == pytest.approx([1.0, 0.5, 1.0])


def test_costmap_clearance_uses_distance_to_nearest_obstacle():
    occupancy = np.zeros((40, 40))
    occupancy[:, 30] = 100
    grid = distance_grid(occupancy, resolution=0.05, origin=[0.0, 0.0], occupied_threshold=50)

    clearance = base_costmap_clearance(grid, np.array([1.0, 1.0, 0.0]), PARAMS)

    assert clearance == pytest.approx(0.5 - 0.3, abs=0.051)


def test_arms_collide_only_when_their_spheres_come_closer_than_the_margin():
    left = [Sphere(np.array([0.5, 0.2, 1.0]), 0.07, False)]
    near = [Sphere(np.array([0.5, 0.02, 1.0]), 0.07, False)]
    far = [Sphere(np.array([0.5, -0.2, 1.0]), 0.07, False)]

    assert arms_collide(left, near, PARAMS)
    assert not arms_collide(left, far, PARAMS)
