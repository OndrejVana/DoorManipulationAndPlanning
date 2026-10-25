import numpy as np

from .door_planner import ArmChecks
from .kinematics import joint_limits, link_spheres, solve_ik
from .tracking import gripper_target_in_base


def has_joint_margin(arm, q, margin):
    lower, upper = joint_limits(arm)
    return bool(np.all(q - lower >= margin) and np.all(upper - q >= margin))


def middle_configuration(arm):
    lower, upper = joint_limits(arm)
    return (lower + upper) / 2.0


def make_arm_checks(
    arm, passive_arm, passive_q, scene, grasp_offset, joint_margin, ik_iterations
):
    passive_spheres = link_spheres(passive_arm, passive_q)

    def solve(base_pose, door_angle, seed):
        target = gripper_target_in_base(scene, door_angle, grasp_offset, base_pose)
        start = middle_configuration(arm) if seed is None else seed
        q, converged = solve_ik(arm, target, start, iterations=ik_iterations)
        return q if converged and has_joint_margin(arm, q, joint_margin) else None

    def spheres(q):
        return link_spheres(arm, q) + passive_spheres

    return ArmChecks(solve, spheres)
