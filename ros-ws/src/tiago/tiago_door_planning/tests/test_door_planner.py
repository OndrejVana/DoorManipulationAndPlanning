import numpy as np

from tiago_door_planning.collision import CollisionParams
from tiago_door_planning.door_planner import ArmChecks, Planner, PlannerParams, PlanningProblem
from tiago_door_planning.door_scene import DoorDimensions, handle_pose, scene_from_door_poses
from tiago_door_planning.geometry import make_pose

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)
COLLISION = CollisionParams(base_radius=0.3, base_center_height=0.15, margin=0.05)
PARAMS = PlannerParams(
    resolution=0.05, heading_bins=16, step_cells=2, samples_per_edge=3,
    angle_step=np.radians(5.0), max_door_steps=2, goal_angle=np.radians(30.0),
    linear_speed=0.2, angular_speed=0.5, door_rate=np.radians(10.0), reverse_penalty=2.0, regrasp_time=15.0,
    max_expansions=3000,
)


def push_scene():
    hinge = make_pose([0.75, -0.55, 0.0], 0.0)
    handle = make_pose([0.75, 0.25, 1.0], 0.0)
    return scene_from_door_poses(hinge, handle, "right", True, DIMENSIONS)


def reach_band_arm(scene, near, far):
    def solve(base_pose, door_angle, seed):
        distance = np.linalg.norm(handle_pose(scene, door_angle)[:2, 3] - base_pose[:2])
        return np.zeros(1) if near <= distance <= far else None
    return ArmChecks(solve, lambda q: [])


def make_planner(near=0.4, far=0.85, params=PARAMS):
    scene = push_scene()
    problem = PlanningProblem(scene, None, np.array([0.0, 0.0, 0.0]), 0.0, np.zeros(1), np.zeros(1),
                              reach_band_arm(scene, near, far), COLLISION)
    return Planner(problem, params)


def test_push_door_reaches_goal_angle():
    steps, _ = make_planner().plan()

    assert steps is not None
    assert steps[-1].angles[-1] >= PARAMS.goal_angle - 1e-9


def test_door_angle_never_decreases():
    steps, _ = make_planner().plan()
    angles = np.concatenate([step.angles for step in steps])

    assert np.all(np.diff(angles) >= -1e-9)


def test_planned_base_poses_keep_handle_in_reach():
    planner = make_planner()
    steps, _ = planner.plan()

    for step in steps:
        for pose, angle in zip(step.poses, step.angles):
            if step.grasped:
                assert planner.problem.arm.solve(pose, angle, None) is not None


def test_unreachable_handle_gives_no_plan():
    steps, expansions = make_planner(near=2.0, far=2.1, params=PARAMS._replace(max_expansions=200)).plan()

    assert steps is None
    assert expansions <= 200
