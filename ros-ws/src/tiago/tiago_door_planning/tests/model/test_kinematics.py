import subprocess

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from tiago_door_planning.model.geometry import local_pose_error  # noqa: E402
from tiago_door_planning.motion.arm_moves import body_clearance  # noqa: E402
from tiago_door_planning.model.kinematics import (  # noqa: E402
    gripper_pose,
    joint_limits,
    forearm_spheres,
    link_spheres,
    load_arm_model,
    solve_ik,
)

SPHERE_LINKS = ["arm_{side}_1_link", "arm_{side}_3_link", "arm_{side}_4_link", "arm_{side}_6_link"]


@pytest.fixture(scope="module")
def urdf_xml():
    try:
        description = subprocess.check_output(["rospack", "find", "tiago_dual_description"]).decode().strip()
        return subprocess.check_output(
            ["rosrun", "xacro", "xacro", description + "/robots/tiago_dual.urdf.xacro"]
        ).decode()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("tiago_dual_description not available")


@pytest.fixture(scope="module")
def arm(urdf_xml):
    return load_arm_model(
        urdf_xml, "left", "hand_{side}_grasping_frame", SPHERE_LINKS, 0.07, 0.1, {"torso_lift_joint": 0.15}
    )


def middle_configuration(arm):
    lower, upper = joint_limits(arm)
    return (lower + upper) / 2.0


def test_model_keeps_only_the_seven_arm_joints(arm):
    assert arm.model.nq == 7


def test_ik_reaches_a_reachable_target(arm):
    target = gripper_pose(arm, middle_configuration(arm) + 0.2)

    q, converged = solve_ik(arm, target, middle_configuration(arm))

    assert converged
    assert np.linalg.norm(local_pose_error(gripper_pose(arm, q), target)) < 1e-3


def test_last_sphere_sits_on_the_gripper(arm):
    q = middle_configuration(arm)

    spheres = link_spheres(arm, q)

    assert spheres[-1].is_gripper
    assert spheres[-1].center == pytest.approx(gripper_pose(arm, q)[:3, 3])


def test_upper_arm_spheres_are_not_gripper(arm):
    spheres = link_spheres(arm, middle_configuration(arm))

    assert not spheres[0].is_gripper


BODY_SPHERES = [
    ["head_2_link", [0.012, 0.04, 0.06], 0.12],
    ["head_2_link", [0.012, 0.04, -0.06], 0.12],
    ["head_1_link", [0.01, 0.0, 0.09], 0.09],
    ["torso_lift_link", [0.05, 0.0, -0.12], 0.19],
]
# Right arm and head recorded on the robot, 2026-11-26 (bag run2): folded home, and the hand on the head while unfolding
TUCKED_Q = [-1.1, 1.468, 2.714, 1.709, -1.571, 1.39, 0.0]
TUCKED_BODY = {"torso_lift_joint": 0.15, "head_1_joint": -0.099, "head_2_joint": -0.071}
ON_HEAD_Q = [0.545, -0.382, 0.916, 2.023, -0.848, 0.31, 0.18]
ON_HEAD_BODY = {"torso_lift_joint": 0.15, "head_1_joint": -0.806, "head_2_joint": -0.067}


def right_arm_with_body(urdf_xml, fixed_positions):
    return load_arm_model(
        urdf_xml,
        "right",
        "hand_{side}_grasping_frame",
        SPHERE_LINKS,
        0.07,
        0.1,
        fixed_positions,
        BODY_SPHERES,
        "arm_{side}_4_link",
    )


def test_tucked_arm_keeps_clear_of_head_and_torso(urdf_xml):
    arm = right_arm_with_body(urdf_xml, TUCKED_BODY)

    clearance = body_clearance(arm, np.array(TUCKED_Q))

    assert clearance > 0.03


def test_hand_on_the_turned_head_is_a_collision(urdf_xml):
    arm = right_arm_with_body(urdf_xml, ON_HEAD_BODY)

    clearance = body_clearance(arm, np.array(ON_HEAD_Q))

    assert clearance < 0.0


def test_no_body_spheres_means_no_body_check(arm):
    assert forearm_spheres(arm, middle_configuration(arm)) == []
    assert body_clearance(arm, middle_configuration(arm)) == np.inf
