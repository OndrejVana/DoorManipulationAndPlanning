import numpy as np
import pytest

from tiago_door_planning.model.collision import CollisionParams
from tiago_door_planning.planning.door_planner import QUICK_ONTO_SPOT_S, SPOT_SIDESTEP, ArmChecks, Planner, PlannerParams, PlanningProblem
from tiago_door_planning.model.door_scene import DoorDimensions, handle_pose, scene_from_door_poses
from tiago_door_planning.model.geometry import make_pose

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)
COLLISION = CollisionParams(base_radius=0.3, base_center_height=0.15, margin=0.05)
PARAMS = PlannerParams(
    resolution=0.05,
    heading_bins=16,
    step_cells=2,
    samples_per_edge=3,
    angle_step=np.radians(5.0),
    max_door_steps=2,
    goal_angle=np.radians(30.0),
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


def push_scene():
    hinge = make_pose([0.75, -0.55, 0.0], 0.0)
    handle = make_pose([0.75, 0.25, 1.0], 0.0)
    return scene_from_door_poses(hinge, handle, "right", True, DIMENSIONS)


def reach_band_arm(scene, near, far, max_angle=np.pi):
    def solve(base_pose, door_angle, seed):
        distance = np.linalg.norm(handle_pose(scene, door_angle)[:2, 3] - base_pose[:2])
        return np.zeros(1) if near <= distance <= far and door_angle <= max_angle + 1e-9 else None

    return ArmChecks(solve, lambda q: [], np.inf, lambda pose, angle, q: True)


def make_planner(near=0.4, far=0.85, params=PARAMS, arms=None, start_grasped=True, regrasp_in_place=True):
    scene = push_scene()
    arms = arms or {"right": reach_band_arm(scene, near, far)}
    problem = PlanningProblem(
        scene, None, np.array([0.0, 0.0, 0.0]), 0.0, np.zeros(1), np.zeros(1), arms, "right", start_grasped, COLLISION
    )
    return Planner(problem, params, regrasp_in_place=regrasp_in_place)


def test_push_door_reaches_goal_angle():
    steps = make_planner().plan()

    assert steps is not None
    assert steps[-1].angles[-1] >= PARAMS.goal_angle - 1e-9


def test_parallel_search_plans_the_same_path():
    serial = make_planner().plan()
    parallel = make_planner(params=PARAMS._replace(workers=2)).plan()

    assert np.array_equal(np.concatenate([step.poses for step in parallel]),
                          np.concatenate([step.poses for step in serial]))


def test_door_angle_never_decreases():
    steps = make_planner().plan()
    angles = np.concatenate([step.angles for step in steps])

    assert np.all(np.diff(angles) >= -1e-9)


def test_planned_base_poses_keep_handle_in_reach():
    planner = make_planner()
    steps = planner.plan()

    for step in steps:
        for pose, angle in zip(step.poses, step.angles):
            if step.grasped:
                assert planner.problem.grips[step.grip].solve(pose, angle, None) is not None


def test_arm_configuration_jumps_never_open_the_door():
    scene = push_scene()
    reach = reach_band_arm(scene, 0.4, 0.85)
    flipping = reach._replace(
        solve=lambda pose, angle, seed: (
            None if reach.solve(pose, angle, seed) is None else np.full(1, 1.0 - (0.0 if seed is None else seed[0]))
        )
    )
    steps = make_planner(arms={"right": flipping}, params=PARAMS._replace(max_expansions=200)).plan()

    assert steps[-1].angles[-1] == 0.0


def test_grasped_edges_last_as_long_as_the_arm_needs():
    scene = push_scene()
    reach = reach_band_arm(scene, 0.4, 0.85)
    swinging = reach._replace(
        solve=lambda pose, angle, seed: (
            None if reach.solve(pose, angle, seed) is None else np.full(1, 0.1 - (0.0 if seed is None else seed[0]))
        )
    )
    steps = make_planner(arms={"right": swinging}, params=PARAMS._replace(arm_speed=0.2)).plan()

    assert steps is not None
    assert min(step.duration for step in steps[1:]) >= 3 * 0.1 / 0.2 - 1e-9


def test_door_angles_after_an_off_grid_start_lie_on_the_grid():
    planner = make_planner()
    planner = Planner(planner.problem._replace(start_angle=np.radians(1.4)), PARAMS)

    assert planner.door_angle(0) == pytest.approx(np.radians(1.4))
    assert planner.door_angle(1) == pytest.approx(np.radians(5.0))
    assert planner.door_angle(3) == pytest.approx(np.radians(15.0))
    assert planner.door_angle(planner.goal_door) == pytest.approx(PARAMS.goal_angle)


def test_unreachable_handle_never_opens_the_door():
    steps = make_planner(near=2.0, far=2.1, params=PARAMS._replace(max_expansions=200)).plan()

    assert steps[-1].angles[-1] == 0.0


def test_released_start_grasps_with_the_arm_that_reaches():
    scene = push_scene()
    arms = {"right": reach_band_arm(scene, 2.0, 2.1), "left": reach_band_arm(scene, 0.4, 0.85)}
    steps = make_planner(arms=arms, start_grasped=False).plan()

    assert steps is not None
    assert not steps[0].grasped
    assert next(step for step in steps if step.grasped).grip == "left"


def test_a_spot_beside_the_base_is_reached_without_stepping_further_sideways_than_it_tolerates():
    released = make_planner(start_grasped=False)
    steps = Planner(released.problem, PARAMS, np.array([[0.0, 0.08, 0.0]])).plan_to_spot()
    poses = [steps[-2].poses[-1]] + list(steps[-1].poses)

    sideways = [abs((after[:2] - before[:2]) @ np.array([-np.sin(before[2]), np.cos(before[2])]))
                for before, after in zip(poses[:-1], poses[1:])]
    assert sum(sideways) <= SPOT_SIDESTEP + 1e-9
    assert np.allclose(poses[-1], [0.0, 0.08, 0.0])


def test_a_grasp_the_arm_cannot_reach_from_its_folded_pose_is_never_planned():
    arm = reach_band_arm(push_scene(), 0.4, 0.85)._replace(reachable=lambda pose, angle, q: False)
    steps = make_planner(arms={"right": arm}, start_grasped=False, params=PARAMS._replace(max_expansions=300)).plan()

    assert not any(step.grasped for step in steps)


def test_a_held_search_without_in_place_regrasps_never_lets_go():
    steps = make_planner(near=0.4, far=0.6, regrasp_in_place=False).plan()

    assert all(step.grasped for step in steps)


def test_the_last_step_onto_a_spot_is_quick_when_one_is_in_reach():
    released = make_planner(start_grasped=False)
    steps = Planner(released.problem, PARAMS, np.array([[0.3, 0.1, 0.0]])).plan_to_spot()

    assert steps[-1].duration <= QUICK_ONTO_SPOT_S
    assert np.allclose(steps[-1].poses[-1], [0.3, 0.1, 0.0])
