import numpy as np

from tiago_door_planning.planning.approach import approach_spots, plan_summary
from tiago_door_planning.planning.door_planner import PathStep

from .test_opening_plan import GOAL, PARAMS, final_angle, make_problem, push_scene, reach_band_arm
from tiago_door_planning.planning.opening_plan import plan_opening


def released_problem(arms=None):
    return make_problem(arms)._replace(start_pose=np.array([-1.0, 0.0, 0.0]), start_grasped=False)


def step(grasped, duration=1.0):
    return PathStep([np.zeros(3)], [0.0], duration, grasped, None, "right")


def test_a_plan_with_one_regrasp_counts_two_grasps():
    steps = [step(True), step(True), step(False), step(False), step(True)]

    assert plan_summary(steps) == (2, 5.0)


def test_every_approach_spot_opens_the_door_from_there():
    problem = released_problem()

    approaches = approach_spots(problem, PARAMS, GOAL, 5)

    assert approaches
    for approach in approaches:
        start = problem._replace(
            start_pose=approach.spot.pose, start_q=approach.spot.q, start_grip=approach.spot.grip, start_grasped=True
        )
        steps, (minimum, _) = plan_opening(start, PARAMS, GOAL)
        assert final_angle(steps) >= minimum - 1e-9


def test_approach_spots_come_fewest_grasps_first():
    approaches = approach_spots(released_problem(), PARAMS, GOAL, 5)

    keys = [(a.grasps, a.duration) for a in approaches]
    assert keys == sorted(keys)


def test_a_door_no_grasp_opens_far_enough_has_no_approach():
    scene = push_scene()
    arms = {"right": reach_band_arm(scene, np.radians(10.0))}

    assert approach_spots(released_problem(arms), PARAMS, GOAL, 5) == []


def test_a_spot_without_an_opening_goal_does_not_hide_the_others():
    problem = released_problem()
    spots = iter([None, "plan"])

    import tiago_door_planning.planning.approach as approach

    real_plan_opening = approach.plan_opening

    def first_spot_has_no_goal(start, params, goal_params, goals):
        if next(spots, "plan") is None:
            return None, None
        return real_plan_opening(start, params, goal_params, goals)

    approach.plan_opening = first_spot_has_no_goal
    try:
        approaches = approach_spots(problem, PARAMS, GOAL, 5)
    finally:
        approach.plan_opening = real_plan_opening

    assert approaches


def test_planning_stops_after_count_approaches():
    assert len(approach_spots(released_problem(), PARAMS, GOAL, 1)) == 1
