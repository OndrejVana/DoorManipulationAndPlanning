import subprocess

import numpy as np
import pytest

pytest.importorskip("pinocchio")

from tiago_door_planning.geometry import local_pose_error  # noqa: E402
from tiago_door_planning.kinematics import (  # noqa: E402
    gripper_pose,
    joint_limits,
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
    return load_arm_model(urdf_xml, "left", "hand_{side}_grasping_frame", SPHERE_LINKS, 0.07, 0.1,
                          {"torso_lift_joint": 0.15})


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
