import collections

import numpy as np

from ..model.door_scene import handle_radius
from .grasp_spots import grasp_pose_exists

GoalParams = collections.namedtuple(
    "GoalParams",
    "preferred_angle pass_width handle_depth reach_min reach_max max_bearing cell_size heading_bins "
    "extension_expansions regrasp_searches workers",
)


def clear_opening(scene, angle, handle_depth):
    """Free width of the doorway for driving straight through, past the leaf and its far-side handle."""
    dims = scene.dimensions
    leaf_edge = dims.width * np.cos(angle) + dims.thickness * np.sin(angle)
    handle_edge = handle_radius(scene) * np.cos(angle) + (
        dims.thickness + handle_depth
    ) * np.sin(angle)
    return dims.width - max(leaf_edge, handle_edge)


def passable_angle(scene, width, handle_depth, angle_step):
    for angle in np.arange(0.0, np.pi / 2.0 + 1e-9, angle_step):
        if clear_opening(scene, angle, handle_depth) >= width:
            return angle
    return None


def goal_range(problem, params, angle_step):
    """(passable angle, cap) where the cap is the largest angle up to preferred_angle with a grasp pose at
    every step from the passable one; None when the door cannot be opened wide enough to pass.
    """
    minimum = passable_angle(
        problem.scene, params.pass_width, params.handle_depth, angle_step
    )
    if minimum is None or minimum > params.preferred_angle:
        return None
    cap = None
    for angle in np.arange(minimum, params.preferred_angle + 1e-9, angle_step):
        if not grasp_pose_exists(problem, angle, params):
            break
        cap = angle
    return None if cap is None else (minimum, cap)
