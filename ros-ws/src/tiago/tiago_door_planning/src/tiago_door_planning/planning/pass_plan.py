import numpy as np

from ..model.door_scene import door_box, door_yaw, wall_boxes
from .door_goal import clear_opening
from .door_planner import Planner
from .grasp_spots import drive_distance, drive_map_in, free_space_around, is_connected


def pass_target(scene, angle, push, distance, handle_depth):
    """Base pose distance past the middle of the gap the leaf at angle leaves free, heading on through: on the side
    the leaf swings to for a push door, the other side for a pull door. Not from where the robot stands, which can
    already be past the door's line."""
    yaw = door_yaw(scene, 0.0)
    along = np.array([np.cos(yaw), np.sin(yaw)])
    swing = scene.opening_sign * np.array([-along[1], along[0]])
    normal = swing if push else -swing
    gap = clear_opening(scene, angle, handle_depth)
    middle = scene.hinge[:2] + along * (scene.dimensions.width - gap / 2.0)
    xy = middle + normal * distance
    return np.array([xy[0], xy[1], np.arctan2(normal[1], normal[0])])


def plan_pass(problem, params, goal_params, distances, push):
    """PathSteps driving the released robot, arm parked at free_q, through the doorway with the leaf held at
    problem.start_angle, to the first of distances past the doorway it can reach; None when none is reachable."""
    walls, door = wall_boxes(problem.scene), door_box(problem.scene, problem.start_angle)
    for distance in distances:
        target = pass_target(
            problem.scene,
            problem.start_angle,
            push,
            distance,
            goal_params.handle_depth,
        )
        free_space = free_space_around(
            problem, walls, door, [target[:2]], goal_params.reach_max, goal_params.cell_size
        )
        if not is_connected(free_space, target[:2]):
            continue
        drive = drive_map_in(free_space, [target[:2]])
        steps = Planner(
            problem,
            params._replace(goal_angle=problem.start_angle),
            np.array([target]),
            lambda xy: drive_distance(drive, xy),
        ).plan_to_spot()
        if steps is not None:
            return steps
    return None
