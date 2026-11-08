import numpy as np

from .door_goal import goal_range
from .door_planner import PathStep, Planner
from ..model.geometry import wrap_angle
from .grasp_spots import drive_distance, drive_map, grasp_spots


def _problem_from_step(problem, step):
    return problem._replace(
        start_pose=step.poses[-1],
        start_angle=step.angles[-1],
        start_q=step.q,
        start_grip=step.grip,
    )


def _released_problem(problem, step):
    return _problem_from_step(problem, step)._replace(start_grasped=False)


def _grasped_problem(problem, step):
    return _problem_from_step(problem, step)._replace(start_grasped=True)


def _final_angle(steps):
    return steps[-1].angles[-1]


def _within_grip_limit(problem, params, goal_angle):
    """A grasp is done at its grip's limit; searching past it only burns expansions."""
    return params._replace(
        goal_angle=min(goal_angle, problem.grips[problem.start_grip].max_angle)
    )


def _open_with_regrasps(problem, params, goal_params):
    steps = Planner(
        problem, _within_grip_limit(problem, params, params.goal_angle)
    ).plan()
    for _ in range(goal_params.regrasp_searches):
        if _final_angle(steps) >= params.goal_angle - 1e-9:
            break
        regrasp = _drive_and_grasp(
            _released_problem(problem, steps[-1]), params, goal_params
        )
        if regrasp is None:
            break
        grasped = _grasped_problem(problem, regrasp[-1])
        more = Planner(
            grasped, _within_grip_limit(grasped, params, params.goal_angle)
        ).plan()
        if _final_angle(more) <= _final_angle(steps) + 1e-9:
            break
        steps = steps + regrasp + more[1:]
    return steps


def _nearest_spot(spots, pose):
    gaps = [
        np.linalg.norm(spot.pose[:2] - pose[:2])
        + abs(wrap_angle(spot.pose[2] - pose[2]))
        for spot in spots
    ]
    return spots[int(np.argmin(gaps))]


def _drive_and_grasp(released, params, goal_params):
    """Steps that drive onto a spot where some grip moves the door one step further, ending with the grasp there."""
    angle = released.start_angle
    spots = grasp_spots(released, angle, angle + params.angle_step, goal_params)
    if not spots:
        return None
    distances = drive_map(released, angle, spots, goal_params)
    spot_poses = np.array([spot.pose for spot in spots])
    drive = Planner(
        released, params, spot_poses, lambda xy: drive_distance(distances, xy)
    ).plan_to_spot()
    if drive is None:
        return None
    spot = _nearest_spot(spots, drive[-1].poses[-1])
    return drive + [
        PathStep(
            [spot.pose], [angle], params.regrasp_time / 2.0, True, spot.q, spot.grip
        )
    ]


def _extend(problem, params, steps, cap, max_expansions):
    last = steps[-1]
    extension_problem = _problem_from_step(problem, last)
    extension_params = _within_grip_limit(extension_problem, params, cap)._replace(
        max_expansions=max_expansions
    )
    if not last.grasped or last.angles[-1] >= extension_params.goal_angle - 1e-9:
        return steps
    extension = Planner(extension_problem, extension_params).plan()
    return steps + extension[1:]


def _open_from_start(problem, params, goal_params):
    """A released robot first drives onto a grasp spot and grasps there."""
    if problem.start_grasped:
        return _open_with_regrasps(problem, params, goal_params)
    regrasp = _drive_and_grasp(problem, params, goal_params)
    if regrasp is None:
        return None
    return (
        regrasp
        + _open_with_regrasps(
            _grasped_problem(problem, regrasp[-1]), params, goal_params
        )[1:]
    )


def plan_opening(problem, params, goal_params, goals=None):
    """Return (steps, goals): goals is (passable angle, cap) or None, steps is None without a passable plan. The
    robot may start holding the handle or released. Goals already known skip their costly search.
    """
    goals = goals or goal_range(problem, goal_params, params.angle_step)
    if goals is None:
        return None, None
    minimum, cap = goals
    steps = _open_from_start(problem, params._replace(goal_angle=minimum), goal_params)
    if steps is None or _final_angle(steps) < minimum - 1e-9:
        return None, goals
    return _extend(problem, params, steps, cap, goal_params.extension_expansions), goals
