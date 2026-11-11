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


def _within_grip_limit(problem, params, goal_angle, sweep_end=np.inf):
    """A grasp is done at its grip's limit or at sweep_end; searching past them only burns expansions."""
    limit = min(problem.grips[problem.start_grip].max_angle, sweep_end)
    return params._replace(goal_angle=min(goal_angle, limit))


def _sweep_end(grasp_angle, params, goal_params):
    """The last grid angle a grasp made at grasp_angle may open the door to: the executor regrasps after max_sweep,
    so a longer grasp would regrasp where the plan never checked."""
    grid_start = np.floor(grasp_angle / params.angle_step + 1e-9) * params.angle_step
    steps = np.floor((grasp_angle + goal_params.max_sweep - grid_start) / params.angle_step + 1e-9)
    return grid_start + steps * params.angle_step


def _grasp_angle(steps):
    """The door angle where the last grasp of the steps was made."""
    index = len(steps) - 1
    while index > 0 and steps[index - 1].grasped:
        index -= 1
    return steps[index].angles[0]


def _open_with_regrasps(problem, params, goal_params, in_place):
    steps = Planner(
        problem,
        _within_grip_limit(
            problem, params, params.goal_angle,
            _sweep_end(problem.start_angle, params, goal_params),
        ),
        regrasp_in_place=in_place,
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
            grasped,
            _within_grip_limit(
                grasped, params, params.goal_angle,
                _sweep_end(grasped.start_angle, params, goal_params),
            ),
            regrasp_in_place=in_place,
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


def _extend(problem, params, goal_params, steps, cap, in_place):
    last = steps[-1]
    extension_problem = _problem_from_step(problem, last)
    sweep_end = _sweep_end(_grasp_angle(steps), params, goal_params)
    extension_params = _within_grip_limit(
        extension_problem, params, cap, sweep_end
    )._replace(max_expansions=goal_params.extension_expansions)
    if not last.grasped or last.angles[-1] >= extension_params.goal_angle - 1e-9:
        return steps
    extension = Planner(
        extension_problem, extension_params, regrasp_in_place=in_place
    ).plan()
    return steps + extension[1:]


def _open_from_start(problem, params, goal_params, in_place):
    """A released robot first drives onto a grasp spot and grasps there."""
    if problem.start_grasped:
        return _open_with_regrasps(problem, params, goal_params, in_place)
    regrasp = _drive_and_grasp(problem, params, goal_params)
    if regrasp is None:
        return None
    return (
        regrasp
        + _open_with_regrasps(
            _grasped_problem(problem, regrasp[-1]), params, goal_params, in_place
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
    steps = _passable_steps(problem, params._replace(goal_angle=minimum), goal_params)
    if steps is None:
        return None, goals
    in_place = not goal_params.spot_regrasps_first
    return _extend(problem, params, goal_params, steps, cap, in_place), goals


def _passable_steps(problem, params, goal_params):
    """A regrasp in place costs a whole release and grasp for a few degrees, so with spot_regrasps_first regrasps go
    through grasp spots first and in place only when that finds no passable plan."""
    for in_place in (False, True) if goal_params.spot_regrasps_first else (True,):
        steps = _open_from_start(problem, params, goal_params, in_place)
        if steps is not None and _final_angle(steps) >= params.goal_angle - 1e-9:
            return steps
    return None
