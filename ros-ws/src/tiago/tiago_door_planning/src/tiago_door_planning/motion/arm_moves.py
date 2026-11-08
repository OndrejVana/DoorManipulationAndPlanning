import collections

import numpy as np

from ..model.collision import (
    FREE,
    GRASP,
    arms_collide,
    robot_clearance,
    robot_collides,
    spheres_clearance,
    spheres_in_world,
)
from ..model.door_scene import door_box, wall_boxes
from .grasp_motion import (
    approach_targets,
    grasp_motions,
    ik_seeds,
    lift_targets,
    line_path,
)
from ..model.kinematics import link_spheres

Surroundings = collections.namedtuple(
    "Surroundings", "scene door_angle base_pose grid passive_spheres"
)


def arm_path_clear(arm, around, qs, mode, collision):
    walls, door = wall_boxes(around.scene), door_box(around.scene, around.door_angle)
    for q in qs:
        active_spheres = link_spheres(arm, q)
        if arms_collide(active_spheres, around.passive_spheres, collision):
            return False
        spheres = spheres_in_world(
            around.base_pose, active_spheres + around.passive_spheres
        )
        if robot_collides(
            walls, door, around.grid, around.base_pose, spheres, mode, collision
        ):
            return False
    return True


def arm_path_clearances(arm, around, qs, mode, collision):
    walls, door = wall_boxes(around.scene), door_box(around.scene, around.door_angle)
    clearances = []
    for q in qs:
        active_spheres = link_spheres(arm, q)
        spheres = spheres_in_world(
            around.base_pose, active_spheres + around.passive_spheres
        )
        clearances.append(
            min(
                spheres_clearance(active_spheres, around.passive_spheres),
                robot_clearance(
                    walls, door, around.grid, around.base_pose, spheres, mode, collision
                ),
            )
        )
    return clearances


def leaves_clear(clearances, margin):
    """A path out of where the arm already is: it may start closer than margin, but never closes in further."""
    limit = min(margin, clearances[0])
    return all(clearance >= limit - 1e-9 for clearance in clearances[1:])


def path_leaves_clear(arm, around, path, collision):
    """The hand on the handle may already be closer to the leaf than the margin, after tracking pushed it there."""
    return leaves_clear(
        arm_path_clearances(arm, around, path, GRASP, collision), collision.margin
    )


def base_path_leaves_clear(arm, around, q, base_poses, collision):
    clearances = [
        arm_path_clearances(
            arm, around._replace(base_pose=pose), [q], GRASP, collision
        )[0]
        for pose in base_poses
    ]
    return leaves_clear(clearances, collision.margin)


def approach_lines(around, offset, grasp):
    """From in front of the handle first, then from half as far where the door or frame is close behind the hand;
    lifting from below reaches it when the robot stands close."""
    scene, angle, base_pose, points = (
        around.scene,
        around.door_angle,
        around.base_pose,
        grasp["approach_points"],
    )
    return [
        approach_targets(scene, angle, offset, base_pose, standoff, points)
        for standoff in (grasp["standoff"], grasp["standoff"] / 2.0)
    ] + [lift_targets(scene, angle, offset, base_pose, grasp["release_drop"], points)]


def grasp_motion(arm, around, offset, current_q, planned_q, unfold_q, grasp, collision):
    """(move, approach) to the handle, or None. The arm configuration the plan grasps with comes first: the opening
    may not continue from another one."""
    seeds = ik_seeds(arm, current_q, grasp["random_seeds"])
    if planned_q is not None:
        seeds = [planned_q] + seeds
    return _clear_grasp_motion(
        arm, around, offset, current_q, seeds, unfold_q, grasp, collision
    )


def grasp_reachable(arm, around, offset, rest_q, grasp_q, unfold_q, grasp, collision):
    """Whether the arm gets from rest_q onto the handle with grasp_q the way grasp_motion moves it."""
    return (
        _clear_grasp_motion(
            arm, around, offset, rest_q, [grasp_q], unfold_q, grasp, collision
        )
        is not None
    )


def _clear_grasp_motion(
    arm, around, offset, current_q, seeds, unfold_q, grasp, collision
):
    for targets in approach_lines(around, offset, grasp):
        candidates = grasp_motions(
            arm,
            targets,
            current_q,
            [unfold_q],
            seeds,
            grasp["joint_margin"],
            grasp["ik_iterations"],
            grasp["joint_step"],
        )
        for move, approach in candidates:
            if arm_path_clear(arm, around, move, FREE, collision) and arm_path_clear(
                arm, around, approach, GRASP, collision
            ):
                return move, approach
    return None


def retrace_path(q, approach, tolerance):
    if approach is None or np.max(np.abs(q - approach[-1])) > tolerance:
        return None
    return [q] + list(reversed(approach[:-1]))


def release_paths(arm, q, approach, back_away, action, grasp):
    """(name, path or None) to try in turn: retracing the way in while the arm still holds the handle as it grasped it,
    else a hook drops out from under the bar and a hand wrapped over the bar (push) backs away from the door, by half
    as much where the frame is close, or lifts its open fingers off the bar; parking takes the hand further away.
    """
    drop = np.array([0.0, 0.0, -grasp["release_drop"]])
    move = back_away if action == "push" else drop
    points, iterations = grasp["approach_points"], grasp["ik_iterations"]
    paths = [
        ("retrace", retrace_path(q, approach, grasp["retrace_tolerance"])),
        ("straight line", line_path(arm, q, move, points, iterations)),
        ("half a line", line_path(arm, q, move / 2.0, points, iterations)),
    ]
    if action == "push":
        paths.append(("lift", line_path(arm, q, -drop, points, iterations)))
    return paths
