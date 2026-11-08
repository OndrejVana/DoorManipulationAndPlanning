import subprocess

import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from tiago_door_planning.model.collision import GRASP, CollisionParams, distance_grid  # noqa: E402
from tiago_door_planning.model.door_scene import DoorDimensions, handle_pose, scene_from_door_poses  # noqa: E402
from tiago_door_planning.model.geometry import local_pose_error, make_pose, quaternion_to_rotation  # noqa: E402
from tiago_door_planning.motion.arm_moves import Surroundings, arm_path_clear, leaves_clear  # noqa: E402
from tiago_door_planning.motion.grasp_motion import (  # noqa: E402
    approach_path,
    approach_targets,
    grasp_configurations,
    grasp_motions,
    grasp_offset,
    ik_seeds,
    lift_targets,
    path_duration,
    retreat_displacement,
    toward_robot,
    unrolled_offset,
    via_path,
)
from tiago_door_planning.model.kinematics import gripper_pose, load_arm_model  # noqa: E402

SPHERE_LINKS = ["arm_{side}_1_link", "arm_{side}_3_link", "arm_{side}_4_link", "arm_{side}_6_link"]
DIMENSIONS = DoorDimensions(width=0.9, thickness=0.06, height=2.0, wall_thickness=0.10, wall_length=1.0)
COLLISION = CollisionParams(base_radius=0.30, base_center_height=0.15, margin=0.03)
BASE_POSE = np.array([-0.05, -0.2, 0.0])
STANDOFF = 0.10


@pytest.fixture(scope="module")
def arm():
    try:
        description = subprocess.check_output(["rospack", "find", "tiago_dual_description"]).decode().strip()
        urdf = subprocess.check_output(
            ["rosrun", "xacro", "xacro", description + "/robots/tiago_dual.urdf.xacro"]
        ).decode()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("tiago_dual_description not available")
    return load_arm_model(
        urdf, "left", "hand_{side}_grasping_frame", SPHERE_LINKS, 0.07, 0.1, {"torso_lift_joint": 0.14}
    )


@pytest.fixture(scope="module")
def scene():
    handle = np.eye(4)
    handle[:3, :3] = quaternion_to_rotation(-0.5012, -0.4977, 0.4994, 0.5018)
    handle[:3, 3] = [0.655, 0.200, 1.052]
    return scene_from_door_poses(make_pose([0.70, -0.55, 0.0], 0.0), handle, "right", True, DIMENSIONS)


@pytest.fixture(scope="module")
def grasp_rotation_offset():
    offset = np.eye(4)
    offset[:3, :3] = pin.rpy.rpyToMatrix(np.pi, np.pi / 2.0, 0.0)
    return offset


@pytest.fixture(scope="module")
def targets(scene, grasp_rotation_offset):
    return approach_targets(scene, 0.0, grasp_rotation_offset, BASE_POSE, STANDOFF, 5)


@pytest.fixture(scope="module")
def grasp_q(arm, targets):
    return next(grasp_configurations(arm, targets[-1], ik_seeds(arm, np.zeros(7), 40), 0.1, 300))


def test_normal_points_to_the_robot_side(scene):
    direction = toward_robot(scene, 0.0, BASE_POSE[:2])

    assert direction @ (BASE_POSE[:2] - handle_pose(scene, 0.0)[:2, 3]) > 0.0


def test_door_offset_moves_grasp_toward_hinge_down_and_robot(scene):
    handle = handle_pose(scene, 0.0)
    offset = grasp_offset(scene, toward_robot(scene, 0.0, BASE_POSE[:2]), np.eye(3), [0.06, 0.03, 0.025])
    shift = (handle @ offset)[:3, 3] - handle[:3, 3]
    to_hinge = scene.hinge[:2] - handle[:2, 3]

    assert np.linalg.norm(shift) == pytest.approx(np.linalg.norm([0.06, 0.03, 0.025]), abs=1e-3)
    assert shift[:2] @ to_hinge > 0.0
    assert shift[2] == pytest.approx(-0.03, abs=1e-3)
    assert shift[:2] @ toward_robot(scene, 0.0, BASE_POSE[:2]) == pytest.approx(0.025, abs=1e-3)


def test_approach_starts_at_standoff_and_ends_at_grasp(targets):
    assert np.isclose(np.linalg.norm(targets[0][:3, 3] - targets[-1][:3, 3]), STANDOFF)
    assert np.allclose(targets[0][:3, :3], targets[-1][:3, :3])


def test_lift_rises_vertically_to_the_same_grasp(scene, grasp_rotation_offset, targets):
    lift = lift_targets(scene, 0.0, grasp_rotation_offset, BASE_POSE, 0.08, 5)

    assert np.allclose(lift[-1], targets[-1])
    assert np.allclose(lift[0][:3, 3], targets[-1][:3, 3] - [0.0, 0.0, 0.08])


def test_approach_path_ends_at_grasp_and_follows_targets(arm, targets, grasp_q):
    path = approach_path(arm, targets, grasp_q, 200)

    assert np.allclose(path[-1], grasp_q)
    for q, target in zip(path, targets):
        assert np.linalg.norm(local_pose_error(gripper_pose(arm, q), target)) < 1e-3


def test_path_duration_follows_the_joint_that_travels_furthest():
    out_and_back = [np.zeros(7), np.full(7, 0.5), np.r_[0.0, np.full(6, 0.5)]]

    assert path_duration(out_and_back, 0.5, 0.1) == pytest.approx(2.0)
    assert path_duration([np.zeros(7), np.zeros(7)], 0.5, 1.0) == pytest.approx(1.0)


def test_via_path_passes_through_via_configuration():
    via = np.full(7, 0.5)
    path = via_path([np.zeros(7), via, np.zeros(7)], 0.1)

    assert any(np.allclose(q, via) for q in path)
    assert np.allclose(path[-1], 0.0)
    assert np.max(np.abs(np.diff(path, axis=0))) <= 0.1 + 1e-9


def test_grasp_motions_end_at_the_handle(arm, targets):
    seeds = ik_seeds(arm, np.zeros(7), 10)
    move, approach = next(grasp_motions(arm, targets, np.zeros(7), [], seeds, 0.1, 300, 0.05))

    assert np.allclose(move[-1], approach[0])
    assert np.linalg.norm(local_pose_error(gripper_pose(arm, approach[-1]), targets[-1])) < 1e-3


def test_approach_is_clear_in_open_scene(arm, scene, targets, grasp_q):
    path = approach_path(arm, targets, grasp_q, 200)

    assert arm_path_clear(arm, Surroundings(scene, 0.0, BASE_POSE, None, []), path, GRASP, COLLISION)


def test_occupied_costmap_blocks_the_path(arm, scene, grasp_q):
    grid = distance_grid(np.full((80, 80), 100), 0.05, [-2.0, -2.0], 50)

    assert not arm_path_clear(arm, Surroundings(scene, 0.0, BASE_POSE, grid, []), [grasp_q], GRASP, COLLISION)


def test_retreat_moves_the_gripper_toward_the_robot(scene):
    base_pose = np.array([BASE_POSE[0], BASE_POSE[1], 0.7])
    displacement = retreat_displacement(scene, 0.0, base_pose, 0.2)
    world = np.array(
        [
            np.cos(0.7) * displacement[0] - np.sin(0.7) * displacement[1],
            np.sin(0.7) * displacement[0] + np.cos(0.7) * displacement[1],
        ]
    )

    assert np.linalg.norm(displacement) == pytest.approx(0.2)
    assert world == pytest.approx(0.2 * toward_robot(scene, 0.0, base_pose[:2]))


def test_unrolled_offset_keeps_the_measured_position_and_the_planned_rotation():
    planned = make_pose([0.0, 0.0, 0.1], 0.0)
    measured = make_pose([0.01, 0.0, 0.12], 0.1)

    offset = unrolled_offset(planned, measured)

    assert offset[:3, 3] == pytest.approx(measured[:3, 3])
    assert offset[:3, :3] == pytest.approx(planned[:3, :3])


def test_a_path_leaving_from_close_to_the_leaf_may_not_close_in():
    assert leaves_clear([0.01, 0.012, 0.03], 0.015)
    assert not leaves_clear([0.01, 0.008, 0.03], 0.015)
    assert not leaves_clear([0.05, 0.02, 0.01], 0.015)
