import numpy as np

from tiago_door_planning.model.collision import CollisionParams, distance_grid
from tiago_door_planning.planning.door_goal import GoalParams, clear_opening, goal_range, passable_angle
from tiago_door_planning.planning.door_planner import ArmChecks, PlanningProblem
from tiago_door_planning.model.door_scene import DoorDimensions, door_box, handle_pose, scene_from_door_poses, wall_boxes
from tiago_door_planning.model.geometry import make_pose, wrap_angle
from tiago_door_planning.planning.grasp_spots import candidate_base_poses, connected_free_space, is_connected

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)
COLLISION = CollisionParams(base_radius=0.27, base_center_height=0.15, margin=0.03)
STEP = np.radians(5.0)
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

    return ArmChecks(solve, lambda q: [], np.inf, lambda pose, angle, q: True)


def make_problem(arm=None, grid=None):
    scene = push_scene()
    arm = arm or reach_band_arm(scene)
    return PlanningProblem(
        scene, grid, np.zeros(3), 0.0, np.zeros(1), np.zeros(1), {"right": arm}, "right", True, COLLISION
    )


def fully_occupied_grid():
    return distance_grid(np.full((160, 160), 100), 0.05, [-4.0, -4.0], 50)


def test_closed_door_has_no_opening():
    assert clear_opening(push_scene(), 0.0, 0.07) <= 0.0


def test_open_door_leaves_width_minus_leaf_and_handle():
    assert np.isclose(clear_opening(push_scene(), np.pi / 2.0, 0.07), 0.9 - 0.04 - 0.07)


def test_wider_robot_needs_wider_angle():
    narrow = passable_angle(push_scene(), 0.5, 0.07, STEP)
    wide = passable_angle(push_scene(), 0.7, 0.07, STEP)

    assert narrow < wide


def test_robot_wider_than_doorway_cannot_pass():
    assert passable_angle(push_scene(), 0.85, 0.07, STEP) is None


def test_candidate_poses_face_the_handle():
    handle = np.array([1.0, 2.0])
    for pose in candidate_base_poses(handle, GOAL):
        bearing = np.arctan2(handle[1] - pose[1], handle[0] - pose[0])
        assert abs(wrap_angle(bearing - pose[2])) <= GOAL.max_bearing + 1e-9
        assert GOAL.reach_min - 1e-9 <= np.linalg.norm(handle - pose[:2]) <= GOAL.reach_max + 1e-9


def test_open_scene_caps_at_preferred_angle():
    minimum, cap = goal_range(make_problem(), GOAL, STEP)

    assert np.isclose(minimum, passable_angle(push_scene(), GOAL.pass_width, GOAL.handle_depth, STEP))
    assert np.isclose(cap, GOAL.preferred_angle)


def test_goal_stops_where_grasp_becomes_impossible():
    scene = push_scene()
    problem = make_problem(arm=reach_band_arm(scene, max_angle=np.radians(80.0)))

    assert np.isclose(goal_range(problem, GOAL, STEP)[1], np.radians(80.0))


def test_goal_is_none_when_passable_angle_has_no_grasp():
    scene = push_scene()
    problem = make_problem(arm=reach_band_arm(scene, max_angle=np.radians(30.0)))

    assert goal_range(problem, GOAL, STEP) is None


def test_goal_is_none_when_no_base_pose_is_free():
    assert goal_range(make_problem(grid=fully_occupied_grid()), GOAL, STEP) is None


def test_free_space_behind_a_wall_is_not_connected():
    occupancy = np.zeros((160, 160))
    occupancy[90, :] = 100
    problem = make_problem(grid=distance_grid(occupancy, 0.05, [-4.0, -4.0], 50))
    walls, door = wall_boxes(problem.scene), door_box(problem.scene, 0.0)
    free_space = connected_free_space(problem, walls, door, np.array([0.0, 0.0]), GOAL)

    assert is_connected(free_space, [0.0, -0.3])
    assert not is_connected(free_space, [0.0, 0.8])
