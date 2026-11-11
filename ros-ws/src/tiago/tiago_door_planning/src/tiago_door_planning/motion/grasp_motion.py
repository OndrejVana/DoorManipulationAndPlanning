"""Grasp geometry: the gripper pose on the handle, straight lines to it, and the arm paths that follow them."""

import numpy as np

from ..model.door_scene import door_yaw, handle_pose
from ..model.geometry import invert, pose2d_to_matrix, rotation_z
from ..model.kinematics import (
    gripper_pose,
    has_joint_margin,
    joint_limits,
    middle_configuration,
    solve_ik,
)


def toward_robot(scene, angle, base_xy):
    """Horizontal door normal at the handle, pointing to the side the robot stands on."""
    yaw = door_yaw(scene, angle)
    normal = np.array([-np.sin(yaw), np.cos(yaw)])
    handle_xy = handle_pose(scene, angle)[:2, 3]
    return normal if normal @ (np.asarray(base_xy) - handle_xy) > 0.0 else -normal


def grasp_offset(scene, normal, rotation, door_offset):
    """Gripper pose in the handle frame; door_offset is (toward the hinge, down, along normal) in metres, and normal
    is the horizontal door normal pointing out of the handle's face."""
    handle = handle_pose(scene, 0.0)
    to_hinge = scene.hinge[:2] - handle[:2, 3]
    to_hinge = to_hinge - (to_hinge @ normal) * normal
    along = to_hinge / np.linalg.norm(to_hinge)
    toward_hinge, down, toward_robot_offset = door_offset
    world_offset = np.array(
        [*(toward_hinge * along + toward_robot_offset * normal), -down]
    )
    offset = np.eye(4)
    offset[:3, :3] = rotation
    offset[:3, 3] = handle[:3, :3].T @ world_offset
    return offset


def unrolled_offset(planned, measured):
    """The measured grasp position with the planned rotation: a hand wrapped around the bar rolls on it freely."""
    offset = planned.copy()
    offset[:3, 3] = measured[:3, 3]
    return offset


def _line_to_grasp(scene, angle, grasp_offset, base_pose, direction, distance, count):
    grasp = handle_pose(scene, angle) @ grasp_offset
    world_to_base = invert(pose2d_to_matrix(base_pose))
    targets = []
    for step in np.linspace(distance, 0.0, count):
        target = grasp.copy()
        target[:3, 3] += step * direction
        targets.append(world_to_base @ target)
    return targets


def approach_targets(scene, angle, grasp_offset, base_pose, standoff, count):
    """Gripper targets in the base frame from the standoff in front of the handle to the grasp."""
    direction = np.array([*toward_robot(scene, angle, base_pose[:2]), 0.0])
    return _line_to_grasp(
        scene, angle, grasp_offset, base_pose, direction, standoff, count
    )


def lift_targets(scene, angle, grasp_offset, base_pose, drop, count):
    """Gripper targets in the base frame rising from below the handle to the grasp."""
    direction = np.array([0.0, 0.0, -1.0])
    return _line_to_grasp(scene, angle, grasp_offset, base_pose, direction, drop, count)


def ik_seeds(arm, current_q, random_count):
    lower, upper = joint_limits(arm)
    rng = np.random.RandomState(0)
    return [current_q, middle_configuration(arm)] + [
        rng.uniform(lower, upper) for _ in range(random_count)
    ]


def grasp_configurations(arm, target, seeds, joint_margin, iterations):
    for seed in seeds:
        q, converged = solve_ik(arm, target, seed, iterations=iterations)
        if converged and has_joint_margin(arm, q, joint_margin):
            yield q


def chained_ik(arm, targets, seed, iterations):
    qs = []
    for target in targets:
        q, converged = solve_ik(arm, target, seed, iterations=iterations)
        if not converged:
            return None
        qs.append(q)
        seed = q
    return qs


def approach_path(arm, targets, grasp_q, iterations):
    """Approach configurations solved backwards from the grasp, so the path ends exactly at grasp_q."""
    backwards = chained_ik(arm, targets[-2::-1], grasp_q, iterations)
    return None if backwards is None else backwards[::-1] + [grasp_q]


def line_path(arm, current_q, displacement, count, iterations):
    """Configurations moving the gripper in a straight line by displacement (base frame), keeping its orientation."""
    start = gripper_pose(arm, current_q)
    targets = []
    for fraction in np.linspace(0.0, 1.0, count)[1:]:
        target = start.copy()
        target[:3, 3] += fraction * np.asarray(displacement)
        targets.append(target)
    path = chained_ik(arm, targets, current_q, iterations)
    return None if path is None else [current_q] + path


def retreat_displacement(scene, angle, base_pose, distance):
    """Base-frame displacement taking the gripper straight away from the door, toward the robot."""
    away = rotation_z(-base_pose[2])[:2, :2] @ toward_robot(scene, angle, base_pose[:2])
    return np.array([*(distance * away), 0.0])


def joint_path(q_start, q_end, max_step):
    count = max(2, int(np.ceil(np.max(np.abs(q_end - q_start)) / max_step)) + 1)
    return [q_start + (q_end - q_start) * t for t in np.linspace(0.0, 1.0, count)]


def waypoint_times(qs, speeds, min_duration):
    """Arrival time at each of qs[1:], every step as quick as its slowest joint allows at speeds (one value, one per
    joint, or one row per step); a path quicker than min_duration is stretched to it."""
    steps = np.abs(np.diff(np.asarray(qs, dtype=float), axis=0)) / speeds
    times = np.cumsum(np.max(steps, axis=1))
    if times[-1] >= min_duration:
        return times
    if times[-1] <= 0.0:
        return min_duration * np.arange(1, len(times) + 1) / len(times)
    return times * min_duration / times[-1]


def via_path(qs, max_step):
    path = [qs[0]]
    for q_start, q_end in zip(qs[:-1], qs[1:]):
        path += joint_path(q_start, q_end, max_step)[1:]
    return path


def grasp_motions(
    arm, targets, current_q, via_qs, seeds, joint_margin, iterations, max_step
):
    """Candidate (move, approach) pairs: each grasp solution reached directly or through one via configuration."""
    for grasp_q in grasp_configurations(
        arm, targets[-1], seeds, joint_margin, iterations
    ):
        approach = approach_path(arm, targets, grasp_q, iterations)
        if approach is None:
            continue
        for via in [[]] + [[q] for q in via_qs]:
            yield via_path([current_q] + via + [approach[0]], max_step), approach
