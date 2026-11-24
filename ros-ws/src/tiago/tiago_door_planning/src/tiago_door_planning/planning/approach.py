import collections

from .grasp_spots import grasp_spots
from .opening_plan import plan_opening

ApproachSpot = collections.namedtuple("ApproachSpot", "spot grasps duration")


def plan_summary(steps):
    """(grasps, duration) of planned steps: every released run ends in a new grasp."""
    grasps = 1 + sum(1 for a, b in zip(steps, steps[1:]) if not a.grasped and b.grasped)
    return grasps, sum(step.duration for step in steps)


def approach_spots(problem, params, goal_params, count, log=lambda text: None):
    """Find up to count approach spots for the closed door."""
    spots = grasp_spots(problem, 0.0, params.angle_step, goal_params, nearest_robot_first=False)
    log("%d grasp spots for the closed door" % len(spots))
    approaches, goals = [], None
    for spot in spots:
        if len(approaches) >= count:
            break
        start = problem._replace(
            start_pose=spot.pose,
            start_angle=0.0,
            start_q=spot.q,
            start_grip=spot.grip,
            start_grasped=True,
        )
        steps, spot_goals = plan_opening(start, params, goal_params, goals)
        if steps is not None:
            goals = spot_goals
            approaches.append(ApproachSpot(spot, *plan_summary(steps)))
        log(
            "spot %s: %s"
            % (
                [round(value, 2) for value in spot.pose],
                "opens the door" if steps is not None
                else "no opening goal" if spot_goals is None
                else "no plan to the goal",
            )
        )
    return sorted(approaches, key=lambda approach: (approach.grasps, approach.duration))
