import numpy as np
import pytest

from tiago_door_planning.model.door_scene import DoorDimensions, handle_pose, scene_from_door_poses
from tiago_door_planning.model.geometry import invert, make_pose, pose2d_to_matrix
from tiago_door_planning.motion.tracking import (
    ABORT,
    DONE,
    JointLimits,
    Measurement,
    TrackingGains,
    arm_velocity,
    nullspace_projector,
    base_command,
    base_tracking_step,
    gripper_target_in_base,
    integrate_joint_command,
    limit_repulsion,
    measure_grasp_offset,
    progress_at_angle,
    progress_rate,
    tracking_step,
    without_roll_on_bar,
)
from tiago_door_planning.motion.trajectory import DoorTrajectory, Reference

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)
GAINS = TrackingGains(
    k_x=1.0,
    k_y=4.0,
    k_theta=2.0,
    k_heading=0.0,
    max_linear=0.3,
    max_angular=0.6,
    arm_gain=2.0,
    damping=0.05,
    nullspace_gain=0.0,
    limit_gain=1.0,
    limit_margin=0.2,
    joint_limits=JointLimits(-np.ones(3), np.ones(3), np.ones(3)),
    gripper_error_slow=0.02,
    gripper_error_stop=0.05,
    gripper_error_abort=0.10,
    base_error_slow=0.05,
    base_error_stop=0.15,
    heading_error_slow=0.09,
    min_base_rate=0.0,
    catch_up_margin=0.0,
)


def make_scene():
    hinge = make_pose([1.0, 0.55, 0.0], 0.0)
    handle = make_pose([1.0, -0.25, 1.0], 0.0)
    return scene_from_door_poses(hinge, handle, "left", False, DIMENSIONS)


def make_trajectory():
    base_poses = np.array([[0.0, 0.0, 0.0], [-0.05, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    return DoorTrajectory(base_poses, np.array([0.0, 0.05, 0.1]), 0.5, "right", np.zeros((3, 3)))


def test_zero_error_gives_feedforward_velocities():
    reference = Reference(np.array([0.0, 0.0, 0.0]), 0.0, 0.2, 0.1, None)

    assert base_command(reference, np.array([0.0, 0.0, 0.0]), 1.0, GAINS) == pytest.approx((0.2, 0.1))


def test_base_lagging_behind_drives_forward():
    reference = Reference(np.array([0.1, 0.0, 0.0]), 0.0, 0.0, 0.0, None)

    linear, _ = base_command(reference, np.array([0.0, 0.0, 0.0]), 1.0, GAINS)

    assert linear > 0.0


def simulate_unicycle(reference_velocity, start_pose, steps=1000, dt=0.05):
    reference_pose = np.zeros(3)
    pose = np.array(start_pose, dtype=float)
    for _ in range(steps):
        reference = Reference(reference_pose.copy(), 0.0, reference_velocity, 0.0, None)
        linear, angular = base_command(reference, pose, 1.0, GAINS)
        pose += dt * np.array([linear * np.cos(pose[2]), linear * np.sin(pose[2]), angular])
        reference_pose[0] += dt * reference_velocity
    return reference_pose, pose


@pytest.mark.parametrize("velocity", [0.1, -0.1])
def test_base_converges_to_reference_forward_and_reversing(velocity):
    reference_pose, pose = simulate_unicycle(velocity, [0.0, 0.05, 0.2])
    assert np.linalg.norm(reference_pose[:2] - pose[:2]) < 0.01
    assert abs(pose[2]) < 0.02


def test_progress_stops_at_large_gripper_error():
    assert progress_rate(0.06, 0.0, GAINS) == 0.0


def test_base_error_alone_never_stops_progress():
    assert progress_rate(0.0, 1.0, GAINS._replace(min_base_rate=0.3)) == pytest.approx(0.3)


def test_small_door_lead_does_not_skip_ahead():
    scene = make_scene()
    trajectory = make_trajectory()
    measurement = measurement_on_target(scene, trajectory, 0.075, np.eye(6, 3))
    measurement = measurement._replace(base_pose=np.zeros(3))
    measurement = measurement._replace(gripper_in_base=handle_pose(scene, 0.01))

    command = tracking_step(trajectory, 0.0, scene, np.eye(4), measurement, GAINS._replace(catch_up_margin=0.035), 0.02)

    assert command.progress < 0.1


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


def test_progress_at_angle_interpolates_reference_angles():
    trajectory = make_trajectory()

    assert progress_at_angle(trajectory, -0.1) == 0.0
    assert progress_at_angle(trajectory, 0.075) == pytest.approx(1.5)
    assert progress_at_angle(trajectory, 0.3) == 2.0


def test_progress_catches_up_with_a_door_that_runs_ahead():
    scene = make_scene()
    trajectory = make_trajectory()
    measurement = measurement_on_target(scene, trajectory, 0.075, np.eye(6, 3))
    measurement = measurement._replace(base_pose=np.array([-0.075, 0.0, 0.0]))
    measurement = measurement._replace(
        gripper_in_base=invert(pose2d_to_matrix(measurement.base_pose)) @ handle_pose(scene, 0.075)
    )

    command = tracking_step(trajectory, 0.0, scene, np.eye(4), measurement, GAINS, 0.02)

    assert command.progress >= 1.5
    assert command.gripper_error < 1e-6


def test_limit_repulsion_pushes_away_from_limits():
    limits = JointLimits(-np.ones(2), np.ones(2), np.ones(2))

    assert limit_repulsion(np.array([-0.95, 0.95]), limits, 0.2) == pytest.approx([0.75, -0.75])


def test_released_step_drives_the_base_and_leaves_the_arm_still():
    trajectory = make_trajectory()
    lagging = trajectory.base_poses[0] + [0.05, 0.0, 0.0]
    measurement = Measurement(lagging, np.zeros(3), np.eye(4), np.eye(6, 3))

    command = base_tracking_step(trajectory, 0.0, measurement, GAINS, 0.02)

    assert command.linear < -0.1
    assert command.joint_velocities == pytest.approx(np.zeros(3))
    assert command.door_angle == trajectory.door_angles[0]


def test_door_seen_past_the_end_finishes_the_segment():
    scene = make_scene()
    trajectory = make_trajectory()
    measurement = measurement_on_target(scene, trajectory, 0.0, np.eye(6, 3))

    command = tracking_step(trajectory, 0.0, scene, np.eye(4), measurement, GAINS, 0.02, door_at_least=0.5)

    assert command.status == DONE
    assert command.linear == 0.0


def test_joint_command_keeps_integrating_when_the_arm_sags():
    q_command = np.zeros(2)
    for _ in range(10):
        q_command = integrate_joint_command(q_command, np.zeros(2), np.array([1.0, -1.0]), 0.02, 0.15)

    assert q_command == pytest.approx([0.15, -0.15])


def test_roll_on_the_bar_is_left_free():
    jacobian = np.eye(6)
    pose_error = np.array([0.0, 0.0, 0.0, 0.3, 0.2, 0.0])

    projected_jacobian, projected_error = without_roll_on_bar(jacobian, pose_error, np.array([1.0, 0.0, 0.0]))

    assert projected_error == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.2, 0.0])
    assert projected_jacobian[3] == pytest.approx(np.zeros(6))


def test_arm_on_its_plan_moves_with_the_planned_velocity():
    planned_velocity = np.array([0.1, -0.2, 0.3])

    velocity = arm_velocity(np.eye(3)[:2], np.zeros(2), np.zeros(3), np.zeros(3), planned_velocity, GAINS)

    assert velocity == pytest.approx(planned_velocity)


def test_pull_to_the_planned_arm_does_not_move_the_gripper():
    jacobian = np.array([[1.0, 0.5, 0.2], [0.0, 1.0, 0.3]])
    gains = GAINS._replace(nullspace_gain=5.0, limit_gain=0.0)
    pulled = arm_velocity(jacobian, np.zeros(2), np.zeros(3), np.array([0.5, -0.4, 0.9]), np.zeros(3), gains)

    assert jacobian @ pulled == pytest.approx(np.zeros(2), abs=1e-9)
    assert np.linalg.norm(pulled) > 0.0
    assert jacobian @ nullspace_projector(jacobian) == pytest.approx(np.zeros((2, 3)), abs=1e-9)


def test_base_turns_onto_a_standing_reference_heading():
    reference = Reference(np.array([0.0, 0.0, 0.3]), 0.0, 0.0, 0.0, None)

    _, angular = base_command(reference, np.array([0.0, 0.0, 0.0]), 0.0, GAINS._replace(k_heading=1.0))

    assert angular == pytest.approx(np.sin(0.3))


def test_progress_waits_for_a_base_left_behind_by_the_door():
    scene = make_scene()
    trajectory = make_trajectory()
    measurement = measurement_on_target(scene, trajectory, 0.075, np.eye(6, 3))
    measurement = measurement._replace(base_pose=np.array([-0.3, 0.0, 0.0]))
    measurement = measurement._replace(
        gripper_in_base=invert(pose2d_to_matrix(measurement.base_pose)) @ handle_pose(scene, 0.075)
    )

    command = tracking_step(trajectory, 0.0, scene, np.eye(4), measurement, GAINS, 0.02)

    assert command.progress < 0.5
    assert command.gripper_error < 1e-6


def test_progress_waits_for_a_base_left_behind_on_a_turn_in_place():
    scene = make_scene()
    turning = make_trajectory()._replace(base_poses=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.2], [0.0, 0.0, -0.4]]))
    measurement = measurement_on_target(scene, turning, 0.075, np.eye(6, 3))
    measurement = measurement._replace(base_pose=np.zeros(3))
    measurement = measurement._replace(gripper_in_base=handle_pose(scene, 0.075))

    command = tracking_step(turning, 0.0, scene, np.eye(4), measurement, GAINS, 0.02)

    assert command.progress < 0.5

