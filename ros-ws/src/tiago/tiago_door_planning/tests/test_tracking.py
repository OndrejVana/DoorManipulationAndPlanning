import numpy as np
import pytest

from tiago_door_planning.door_scene import DoorDimensions, handle_pose, scene_from_door_poses
from tiago_door_planning.geometry import invert, make_pose, pose2d_to_matrix
from tiago_door_planning.tracking import (
    ABORT,
    DONE,
    JointLimits,
    Measurement,
    TrackingGains,
    base_command,
    gripper_target_in_base,
    limit_repulsion,
    measure_grasp_offset,
    progress_rate,
    tracking_step,
)
from tiago_door_planning.trajectory import DoorTrajectory, Reference

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)
GAINS = TrackingGains(
    k_x=1.0, k_y=4.0, k_theta=2.0, max_linear=0.3, max_angular=0.6,
    arm_gain=2.0, damping=0.05, nullspace_gain=0.0, limit_gain=1.0, limit_margin=0.2,
    joint_limits=JointLimits(-np.ones(3), np.ones(3), np.ones(3)),
    gripper_error_slow=0.02, gripper_error_stop=0.05, gripper_error_abort=0.10,
    base_error_slow=0.05, base_error_stop=0.15,
)


def make_scene():
    hinge = make_pose([1.0, 0.55, 0.0], 0.0)
    handle = make_pose([1.0, -0.25, 1.0], 0.0)
    return scene_from_door_poses(hinge, handle, "left", False, DIMENSIONS)


def make_trajectory():
    base_poses = np.array([[0.0, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    return DoorTrajectory(base_poses, np.array([0.0, 0.05, 0.1]), 0.5, "right", np.zeros(3))


def test_zero_error_gives_feedforward_velocities():
    reference = Reference(np.array([0.0, 0.0, 0.0]), 0.0, 0.2, 0.1)

    assert base_command(reference, np.array([0.0, 0.0, 0.0]), 1.0, GAINS) == pytest.approx((0.2, 0.1))


def test_base_lagging_behind_drives_forward():
    reference = Reference(np.array([0.1, 0.0, 0.0]), 0.0, 0.0, 0.0)

    linear, _ = base_command(reference, np.array([0.0, 0.0, 0.0]), 1.0, GAINS)

    assert linear > 0.0


def test_progress_stops_at_large_gripper_error():
    assert progress_rate(0.06, 0.0, GAINS) == 0.0


def test_progress_runs_at_full_rate_when_tracking_well():
    assert progress_rate(0.0, 0.0, GAINS) == 1.0


def test_grasp_offset_reproduces_measured_gripper():
    scene = make_scene()
    base_pose = np.array([0.2, -0.1, 0.1])
    gripper_in_base = make_pose([0.6, -0.2, 1.0], 0.3)

    offset = measure_grasp_offset(scene, 0.0, base_pose, gripper_in_base)

    assert gripper_target_in_base(scene, 0.0, offset, base_pose) == pytest.approx(gripper_in_base)


def test_target_compensates_base_offset():
    scene = make_scene()
    offset = np.eye(4)
    planned = gripper_target_in_base(scene, 0.0, offset, np.array([0.0, 0.0, 0.0]))

    shifted = gripper_target_in_base(scene, 0.0, offset, np.array([0.05, 0.0, 0.0]))

    assert shifted[0, 3] == pytest.approx(planned[0, 3] - 0.05)


def measurement_on_target(scene, trajectory, progress_angle, jacobian):
    base_pose = trajectory.base_poses[0]
    gripper = invert(pose2d_to_matrix(base_pose)) @ handle_pose(scene, progress_angle)
    return Measurement(base_pose, np.zeros(3), gripper, jacobian)


def test_step_aborts_when_gripper_lost_the_handle():
    scene = make_scene()
    trajectory = make_trajectory()
    measurement = measurement_on_target(scene, trajectory, 0.0, np.eye(6, 3))
    lost = measurement._replace(gripper_in_base=measurement.gripper_in_base @ make_pose([0.3, 0, 0], 0.0))

    command = tracking_step(trajectory, 0.0, scene, np.eye(4), lost, GAINS, 0.02)

    assert command.status == ABORT


def test_step_reaches_done_at_trajectory_end():
    scene = make_scene()
    trajectory = make_trajectory()
    measurement = measurement_on_target(scene, trajectory, 0.1, np.eye(6, 3))
    measurement = measurement._replace(base_pose=trajectory.base_poses[-1])
    measurement = measurement._replace(
        gripper_in_base=invert(pose2d_to_matrix(measurement.base_pose)) @ handle_pose(scene, 0.1)
    )

    command = tracking_step(trajectory, 1.99, scene, np.eye(4), measurement, GAINS, 0.02)

    assert command.status == DONE


def test_limit_repulsion_is_zero_inside_margin():
    limits = JointLimits(-np.ones(2), np.ones(2), np.ones(2))

    assert limit_repulsion(np.zeros(2), limits, 0.2) == pytest.approx([0.0, 0.0])


def test_limit_repulsion_pushes_away_from_limits():
    limits = JointLimits(-np.ones(2), np.ones(2), np.ones(2))

    assert limit_repulsion(np.array([-0.95, 0.95]), limits, 0.2) == pytest.approx([0.75, -0.75])
