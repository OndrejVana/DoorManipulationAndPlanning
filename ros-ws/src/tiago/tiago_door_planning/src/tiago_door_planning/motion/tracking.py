import collections
import numpy as np

from ..model.door_scene import door_angle_at, handle_pose
from ..model.geometry import (
    damped_pseudo_inverse,
    invert,
    local_pose_error,
    pose2d_to_matrix,
    rotation_z,
    wrap_angle,
)
from .trajectory import last_progress, reference_at

TrackingGains = collections.namedtuple(
    "TrackingGains",
    "k_x k_y k_theta k_heading max_linear max_angular arm_gain damping nullspace_gain limit_gain limit_margin "
    "joint_limits gripper_error_slow gripper_error_stop gripper_error_abort base_error_slow base_error_stop "
    "heading_error_slow min_base_rate catch_up_margin",
)

JointLimits = collections.namedtuple("JointLimits", "lower upper velocity")

Measurement = collections.namedtuple(
    "Measurement", "base_pose q gripper_in_base jacobian"
)

TrackingCommand = collections.namedtuple(
    "TrackingCommand",
    "linear angular joint_velocities progress gripper_error base_error door_angle status",
)

OK = "ok"
DONE = "done"
ABORT = "abort"


def slowdown_factor(error, error_slow, error_stop):
    return float(np.clip((error_stop - error) / (error_stop - error_slow), 0.0, 1.0))


def base_rate(base_error, gains):
    return max(
        slowdown_factor(base_error, gains.base_error_slow, gains.base_error_stop),
        gains.min_base_rate,
    )


def progress_rate(gripper_error, base_error, gains):
    return min(
        slowdown_factor(
            gripper_error, gains.gripper_error_slow, gains.gripper_error_stop
        ),
        base_rate(base_error, gains),
    )


def next_progress(trajectory, progress, rate, dt):
    return min(
        progress + rate * dt / trajectory.sample_period, last_progress(trajectory)
    )


def door_angle_from_gripper(scene, grasp_offset, base_pose, gripper_in_base):
    """Door angle implied by the gripper holding the handle."""
    handle = pose2d_to_matrix(base_pose) @ gripper_in_base @ invert(grasp_offset)
    return door_angle_at(scene, handle[:2, 3])


def progress_at_angle(trajectory, angle):
    """Earliest progress at which the reference door angle reaches angle."""
    angles = trajectory.door_angles
    index = int(np.searchsorted(angles, angle))
    if index == 0:
        return 0.0
    if index >= len(angles):
        return last_progress(trajectory)
    lower, upper = angles[index - 1], angles[index]
    return index - 1 + (angle - lower) / (upper - lower)


def base_distance(reference, measurement):
    return float(np.linalg.norm(reference.base_pose[:2] - measurement.base_pose[:2]))


def base_tracking_error(reference_pose, measured_pose):
    """Reference minus measured pose, expressed in the measured base frame."""
    offset = rotation_z(-measured_pose[2])[:2, :2] @ (
        reference_pose[:2] - measured_pose[:2]
    )
    return offset[0], offset[1], wrap_angle(reference_pose[2] - measured_pose[2])


def base_command(reference, measured_pose, rate, gains):
    """Kanayama unicycle tracking law; the heading term uses |v| so it also converges when reversing, and k_heading
    turns the base onto the reference heading while the reference stands still."""
    error_x, error_y, error_theta = base_tracking_error(
        reference.base_pose, measured_pose
    )
    linear_ff, angular_ff = rate * reference.linear, rate * reference.angular
    linear = linear_ff * np.cos(error_theta) + gains.k_x * error_x
    angular = (
        angular_ff
        + gains.k_y * linear_ff * error_y
        + gains.k_theta * abs(linear_ff) * np.sin(error_theta)
        + gains.k_heading * np.sin(error_theta)
    )
    return (
        float(np.clip(linear, -gains.max_linear, gains.max_linear)),
        float(np.clip(angular, -gains.max_angular, gains.max_angular)),
    )


def measure_grasp_offset(scene, door_angle, base_pose, gripper_in_base):
    """Handle-to-gripper transform of the grasp actually achieved."""
    gripper_in_world = pose2d_to_matrix(base_pose) @ gripper_in_base
    return invert(handle_pose(scene, door_angle)) @ gripper_in_world


def gripper_target_in_base(scene, door_angle, grasp_offset, base_pose):
    return (
        invert(pose2d_to_matrix(base_pose))
        @ handle_pose(scene, door_angle)
        @ grasp_offset
    )


def limit_repulsion(q, limits, margin):
    """Zero away from the limits, growing to +1 (-1) at the lower (upper) limit."""
    near_lower = np.clip((limits.lower + margin - q) / margin, 0.0, 1.0)
    near_upper = np.clip((q - (limits.upper - margin)) / margin, 0.0, 1.0)
    return near_lower - near_upper


def bar_axis_in_gripper(grasp_offset):
    """The handle bar runs along the handle frame's x axis."""
    return grasp_offset[:3, :3].T[:, 0]


def without_roll_on_bar(jacobian, pose_error, bar_axis):
    """A hand wrapped around the bar rolls on it freely, so that rotation is left to the arm's null space."""
    projection = np.eye(6)
    projection[3:, 3:] -= np.outer(bar_axis, bar_axis)
    return projection @ jacobian, projection @ pose_error


def nullspace_projector(jacobian):
    """Exact, unlike one built from the damped inverse, which lets the secondary motion move the gripper."""
    return np.eye(jacobian.shape[1]) - np.linalg.pinv(jacobian, rcond=1e-3) @ jacobian


def arm_velocity(jacobian, pose_error, q, planned_q, planned_velocity, gains):
    """The arm moves along the plan; the pose error only corrects what the plan misses."""
    pseudo_inverse = damped_pseudo_inverse(jacobian, gains.damping)
    task = planned_velocity + pseudo_inverse @ (gains.arm_gain * pose_error)
    nullspace = nullspace_projector(jacobian)
    secondary = gains.nullspace_gain * (
        planned_q - q
    ) + gains.limit_gain * limit_repulsion(q, gains.joint_limits, gains.limit_margin)
    return within_velocity_limits(
        task + nullspace @ secondary, gains.joint_limits.velocity
    )


def within_velocity_limits(velocity, limit):
    """Scaled as a whole, since clipping joints one by one would move the gripper off its path."""
    return velocity / max(1.0, float(np.max(np.abs(velocity) / limit)))


def integrate_joint_command(q_command, q_measured, joint_velocities, dt, max_lead):
    """Next commanded joint positions, integrated rather than re-anchored on the measurement: a position controller
    sagging under load then still gets an error large enough to move, bounded to max_lead ahead of the arm.
    """
    lead = q_command + joint_velocities * dt - q_measured
    return q_measured + np.clip(lead, -max_lead, max_lead)


def standing_still(measurement, progress, door_angle, status, gripper_error=0.0):
    return TrackingCommand(
        0.0,
        0.0,
        np.zeros(len(measurement.q)),
        progress,
        gripper_error,
        0.0,
        door_angle,
        status,
    )


def caught_up_progress(trajectory, progress, door_angle, measurement, gains):
    """Progress jumps to a door that ran ahead only as far as the base can follow; the arm keeps up with the door
    on its own meanwhile."""
    caught_up = progress_at_angle(trajectory, door_angle)
    if caught_up <= progress or not base_can_follow(
        reference_at(trajectory, caught_up), measurement, gains
    ):
        return progress
    return caught_up


def base_can_follow(reference, measurement, gains):
    """Near the reference in position and in heading: a turn in place leaves the position where it was."""
    turn = abs(wrap_angle(reference.base_pose[2] - measurement.base_pose[2]))
    return (
        base_distance(reference, measurement) <= gains.base_error_slow
        and turn <= gains.heading_error_slow
    )


def tracking_step(
    trajectory,
    progress,
    scene,
    grasp_offset,
    measurement,
    gains,
    dt,
    door_at_least=None,
):
    """The door is passive: when it runs ahead of the reference, progress jumps to the angle the gripper implies,
    and the segment ends when the laser shows the door already past its end."""
    if (
        door_at_least is not None
        and door_at_least - gains.catch_up_margin >= trajectory.door_angles[-1]
    ):
        return standing_still(
            measurement, last_progress(trajectory), door_at_least, DONE
        )
    door_angle = door_angle_from_gripper(
        scene, grasp_offset, measurement.base_pose, measurement.gripper_in_base
    )
    progress = caught_up_progress(
        trajectory, progress, door_angle - gains.catch_up_margin, measurement, gains
    )
    reference = reference_at(trajectory, progress)
    target = gripper_target_in_base(
        scene,
        max(reference.door_angle, door_angle - gains.catch_up_margin),
        grasp_offset,
        measurement.base_pose,
    )
    pose_error = local_pose_error(measurement.gripper_in_base, target)
    gripper_error = float(np.linalg.norm(pose_error[:3]))
    if gripper_error > gains.gripper_error_abort:
        return standing_still(measurement, progress, door_angle, ABORT, gripper_error)

    base_error = base_distance(reference, measurement)
    rate = progress_rate(gripper_error, base_error, gains)
    linear, angular = base_command(reference, measurement.base_pose, rate, gains)
    next_step = next_progress(trajectory, progress, rate, dt)
    planned_velocity = (
        reference_at(trajectory, next_step).arm_q - reference.arm_q
    ) / dt
    jacobian, task_error = without_roll_on_bar(
        measurement.jacobian, pose_error, bar_axis_in_gripper(grasp_offset)
    )
    joint_velocities = arm_velocity(
        jacobian, task_error, measurement.q, reference.arm_q, planned_velocity, gains
    )
    progress = next_step
    status = DONE if progress >= last_progress(trajectory) else OK
    return TrackingCommand(
        linear,
        angular,
        joint_velocities,
        progress,
        gripper_error,
        base_error,
        door_angle,
        status,
    )


def base_tracking_step(trajectory, progress, measurement, gains, dt):
    """Released hand: the base follows its reference alone and the door stays where it was left."""
    reference = reference_at(trajectory, progress)
    base_error = base_distance(reference, measurement)
    rate = base_rate(base_error, gains)
    linear, angular = base_command(reference, measurement.base_pose, rate, gains)
    progress = next_progress(trajectory, progress, rate, dt)
    status = DONE if progress >= last_progress(trajectory) else OK
    return TrackingCommand(
        linear,
        angular,
        np.zeros(len(measurement.q)),
        progress,
        0.0,
        base_error,
        reference.door_angle,
        status,
    )
