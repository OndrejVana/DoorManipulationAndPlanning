import numpy as np

from tiago_door_planning.model.collision import distance_grid
from tiago_door_planning.model.door_scene import door_box, wall_boxes
from tiago_door_planning.model.geometry import points_to_boxes_distances
from tiago_door_planning.planning.door_goal import clear_opening
from tiago_door_planning.planning.pass_plan import pass_target, plan_pass

from .test_opening_plan import COLLISION, GOAL, PARAMS, make_problem, push_scene

OPEN = np.radians(90.0)


def pass_problem(angle, start_pose, grid=None):
    return make_problem()._replace(
        start_pose=np.asarray(start_pose, dtype=float),
        start_angle=angle,
        start_grasped=False,
        grid=grid,
    )


def wall_grid(scene):
    """A costmap wall along the closed door's line, so the doorway is the only way through."""
    resolution, origin = 0.05, np.array([-2.0, -3.0])
    occupancy = np.zeros((120, 100))
    column = int(round((scene.hinge[0] + 0.05 - origin[0]) / resolution))
    rows = np.arange(occupancy.shape[0])
    ys = origin[1] + rows * resolution
    lo, hi = sorted([scene.hinge[1], scene.handle_closed[1, 3] + 0.4])
    occupancy[(ys < lo - 0.1) | (ys > hi + 0.1), column] = 100
    return distance_grid(occupancy, resolution, origin, 100)


def test_the_target_lies_past_the_gap_on_the_far_side():
    scene = push_scene()
    target = pass_target(scene, OPEN, [0.0, 0.0], 1.0, GOAL.handle_depth)

    assert target[0] > scene.hinge[0] + 0.9
    assert np.isclose(target[2], 0.0)


def test_the_robot_drives_through_the_open_doorway_clear_of_the_leaf():
    scene = push_scene()
    problem = pass_problem(OPEN, [0.1, -0.1, 0.0])

    steps = plan_pass(problem, PARAMS, GOAL, [1.0])

    assert steps is not None
    target = pass_target(scene, OPEN, problem.start_pose[:2], 1.0, GOAL.handle_depth)
    assert np.allclose(steps[-1].poses[-1], target, atol=1e-6)
    poses = np.vstack([step.poses for step in steps])
    centers = np.column_stack([poses[:, :2], np.full(len(poses), COLLISION.base_center_height)])
    clearance = points_to_boxes_distances(centers, wall_boxes(scene) + [door_box(scene, OPEN)])
    assert np.all(clearance - COLLISION.base_radius >= COLLISION.margin - 1e-9)
    assert clear_opening(scene, OPEN, GOAL.handle_depth) >= 0.6


def test_a_closed_door_cannot_be_passed():
    scene = push_scene()
    problem = pass_problem(0.0, [0.1, -0.1, 0.0], wall_grid(scene))

    assert plan_pass(problem, PARAMS, GOAL, [1.0]) is None
