import numpy as np

from tiago_door_planning.model.collision import CollisionParams
from tiago_door_planning.planning.door_goal import GoalParams
from tiago_door_planning.planning.door_planner import ArmChecks, Grip, PlannerParams, PlanningProblem
from tiago_door_planning.model.door_scene import DoorDimensions, back_scene, door_angle_at, handle_pose, scene_from_door_poses
from tiago_door_planning.model.geometry import make_pose
from tiago_door_planning.planning.opening_plan import plan_opening

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)
COLLISION = CollisionParams(base_radius=0.27, base_center_height=0.15, margin=0.03)
PARAMS = PlannerParams(
    resolution=0.05,
    heading_bins=16,
    step_cells=2,
    samples_per_edge=3,
    angle_step=np.radians(5.0),
    max_door_steps=2,
    goal_angle=0.0,
    linear_speed=0.2,
    angular_speed=0.5,
    door_rate=np.radians(10.0),
    regrasp_time=15.0,
    max_expansions=3000,
    stall_expansions=300,
    max_joint_step=0.5,
    arm_speed=1.0,
    workers=1,
)
GOAL = GoalParams(
    preferred_angle=np.radians(90.0),
    pass_width=0.6,
    handle_depth=0.07,
    reach_min=0.35,
    reach_max=0.95,
    max_bearing=np.radians(100.0),
    cell_size=0.1,
    heading_bins=8,
    extension_expansions=300,
    regrasp_searches=2,
    spot_regrasps_first=True,
    max_sweep=np.pi,
    workers=1,
)


def push_scene():
    hinge = make_pose([0.75, -0.55, 0.0], 0.0)
    handle = make_pose([0.75, 0.2, 1.0], 0.0)
    return scene_from_door_poses(hinge, handle, "right", True, DIMENSIONS)


def reach_band_arm(scene, max_angle=np.pi):
    def solve(base_pose, door_angle, seed):
        distance = np.linalg.norm(handle_pose(scene, door_angle)[:2, 3] - base_pose[:2])
        return np.zeros(1) if 0.4 <= distance <= 0.85 and door_angle <= max_angle + 1e-9 else None

    return ArmChecks(solve, lambda q: [], max_angle, lambda pose, angle, q: True)


def pull_scene():
    hinge = make_pose([0.75, 0.55, 0.0], 0.0)
    handle = make_pose([0.75, -0.2, 1.0], 0.0)
    return scene_from_door_poses(hinge, handle, "left", False, DIMENSIONS)


def far_side_arm(scene):
    """Reaches the back handle only from the side of the leaf that faced away from the robot."""
    reach = reach_band_arm(back_scene(scene))

    def solve(base_pose, door_angle, seed):
        on_far_side = door_angle_at(scene, base_pose[:2]) < door_angle - np.radians(5.0)
        return reach.solve(base_pose, door_angle, seed) if on_far_side else None

    return ArmChecks(solve, lambda q: [], np.inf, lambda pose, angle, q: True)


def make_problem(arms=None, scene=None, start_grip="right"):
    scene = scene or push_scene()
    arms = arms or {"right": reach_band_arm(scene)}
    return PlanningProblem(scene, None, np.zeros(3), 0.0, np.zeros(1), np.zeros(1), arms, start_grip, True, COLLISION)


def final_angle(steps):
    return steps[-1].angles[-1]


def test_open_scene_opens_to_the_cap():
    steps, (_, cap) = plan_opening(make_problem(), PARAMS, GOAL)

    assert np.isclose(final_angle(steps), cap)


def test_known_goals_are_kept():
    _, (minimum, _) = plan_opening(make_problem(), PARAMS, GOAL)
    steps, goals = plan_opening(make_problem(), PARAMS, GOAL, (minimum, minimum))

    assert goals == (minimum, minimum)
    assert np.isclose(final_angle(steps), minimum)


def test_tiny_extension_budget_still_passes():
    steps, (minimum, _) = plan_opening(make_problem(), PARAMS, GOAL._replace(extension_expansions=1))

    assert final_angle(steps) >= minimum - 1e-9


def test_door_angle_never_decreases_across_stages():
    steps, _ = plan_opening(make_problem(), PARAMS, GOAL)
    angles = np.concatenate([step.angles for step in steps])

    assert np.all(np.diff(angles) >= -1e-9)


def test_hands_over_to_the_other_arm_when_the_first_runs_out_of_reach():
    scene = push_scene()
    arms = {"right": reach_band_arm(scene, np.radians(15.0)), "left": reach_band_arm(scene)}

    steps, (minimum, _) = plan_opening(make_problem(arms), PARAMS, GOAL)

    assert steps[0].grip == "right"
    assert steps[-1].grip == "left"
    assert final_angle(steps) >= minimum - 1e-9


def test_pull_door_is_pushed_on_from_its_far_side_past_the_pull_limit():
    scene = pull_scene()
    front, back = Grip("right", "front"), Grip("right", "back")
    grips = {front: reach_band_arm(scene, np.radians(45.0)), back: far_side_arm(scene)}

    steps, (minimum, _) = plan_opening(make_problem(grips, scene, front), PARAMS, GOAL)

    assert steps[0].grip == front
    assert steps[-1].grip == back
    assert final_angle(steps) >= minimum - 1e-9


def test_released_robot_drives_to_a_grasp_and_opens():
    problem = make_problem()._replace(start_grasped=False)

    steps, (minimum, _) = plan_opening(problem, PARAMS, GOAL)

    assert not steps[0].grasped
    assert final_angle(steps) >= minimum - 1e-9


def grasp_sweeps(steps):
    sweeps, start = [], None
    for step in steps:
        if not step.grasped:
            start = None
            continue
        start = step.angles[0] if start is None else start
        sweeps.append(step.angles[-1] - start)
    return sweeps


def test_no_grasp_opens_the_door_further_than_the_executor_sweeps():
    max_sweep = np.radians(25.0)

    steps, (minimum, _) = plan_opening(make_problem(), PARAMS, GOAL._replace(max_sweep=max_sweep, regrasp_searches=4))

    assert max(grasp_sweeps(steps)) <= max_sweep + 1e-9
    assert final_angle(steps) >= minimum - 1e-9
