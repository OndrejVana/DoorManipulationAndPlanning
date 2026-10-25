import collections
import numpy as np

from .door_scene import handle_pose
from .geometry import (
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
    [
        "k_x",
        "k_y",
        "k_theta",
        "max_linear",
        "max_angular",
        "arm_gain",
        "damping",
        "nullspace_gain",
        "limit_gain",
        "limit_margin",
        "joint_limits",
        "gripper_error_slow",
        "gripper_error_stop",
        "gripper_error_abort",
        "base_error_slow",
        "base_error_stop",
    ],
)

JointLimits = collections.namedtuple("JointLimits", ["lower", "upper", "velocity"])

Measurement = collections.namedtuple(
    "Measurement", ["base_pose", "q", "gripper_in_base", "jacobian"]
)

TrackingCommand = collections.namedtuple(
    "TrackingCommand",
    ["linear", "angular", "joint_velocities", "progress", "gripper_error", "status"],
)

OK = "ok"
DONE = "done"
ABORT = "abort"


def slowdown_factor(error, error_slow, error_stop):
    return float(np.clip((error_stop - error) / (error_stop - error_slow), 0.0, 1.0))


def progress_rate(gripper_error, base_error, gains):
    return min(
        slowdown_factor(
            gripper_error, gains.gripper_error_slow, gains.gripper_error_stop
        ),
        slowdown_factor(base_error, gains.base_error_slow, gains.base_error_stop),
    )


def next_progress(trajectory, progress, rate, dt):
    return min(
        progress + rate * dt / trajectory.sample_period, last_progress(trajectory)
    )


def base_tracking_error(reference_pose, measured_pose):
    """Reference minus measured pose, expressed in the measured base frame."""
    offset = rotation_z(-measured_pose[2])[:2, :2] @ (
        reference_pose[:2] - measured_pose[:2]
    )
    return offset[0], offset[1], wrap_angle(reference_pose[2] - measured_pose[2])


def base_command(reference, measured_pose, rate, gains):
    """Kanayama unicycle tracking law: feedforward plus position and heading feedback."""
    error_x, error_y, error_theta = base_tracking_error(
        reference.base_pose, measured_pose
    )
    linear_ff, angular_ff = rate * reference.linear, rate * reference.angular
    linear = linear_ff * np.cos(error_theta) + gains.k_x * error_x
    angular = angular_ff + linear_ff * (
        gains.k_y * error_y + gains.k_theta * np.sin(error_theta)
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


def arm_velocity(jacobian, pose_error, q, elbow_hint, gains):
    pseudo_inverse = damped_pseudo_inverse(jacobian, gains.damping)
    task = pseudo_inverse @ (gains.arm_gain * pose_error)
    nullspace = np.eye(len(q)) - pseudo_inverse @ jacobian
    secondary = gains.nullspace_gain * (
        elbow_hint - q
    ) + gains.limit_gain * limit_repulsion(q, gains.joint_limits, gains.limit_margin)
    velocity_limit = gains.joint_limits.velocity
    return np.clip(task + nullspace @ secondary, -velocity_limit, velocity_limit)


def stop_command(measurement, progress, gripper_error):
    return TrackingCommand(
        0.0, 0.0, np.zeros(len(measurement.q)), progress, gripper_error, ABORT
    )


def tracking_step(trajectory, progress, scene, grasp_offset, measurement, gains, dt):
    reference = reference_at(trajectory, progress)
    target = gripper_target_in_base(
        scene, reference.door_angle, grasp_offset, measurement.base_pose
    )
    pose_error = local_pose_error(measurement.gripper_in_base, target)
    gripper_error = float(np.linalg.norm(pose_error[:3]))
    if gripper_error > gains.gripper_error_abort:
        return stop_command(measurement, progress, gripper_error)

    base_error = float(
        np.linalg.norm(reference.base_pose[:2] - measurement.base_pose[:2])
    )
    rate = progress_rate(gripper_error, base_error, gains)
    linear, angular = base_command(reference, measurement.base_pose, rate, gains)
    joint_velocities = arm_velocity(
        measurement.jacobian, pose_error, measurement.q, trajectory.elbow_hint, gains
    )
    progress = next_progress(trajectory, progress, rate, dt)
    status = DONE if progress >= last_progress(trajectory) else OK
    return TrackingCommand(
        linear, angular, joint_velocities, progress, gripper_error, status
    )
