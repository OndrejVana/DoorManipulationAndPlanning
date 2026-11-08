import collections
import numpy as np

from ..model.geometry import wrap_angle

SAMPLE_PERIOD = 0.1

DoorTrajectory = collections.namedtuple(
    "DoorTrajectory", "base_poses door_angles sample_period side arm_qs"
)

Reference = collections.namedtuple(
    "Reference", "base_pose door_angle linear angular arm_q"
)


def last_progress(trajectory):
    return float(len(trajectory.door_angles) - 1)


def _interpolate_pose(start, end, fraction):
    xy = start[:2] + fraction * (end[:2] - start[:2])
    yaw = start[2] + fraction * wrap_angle(end[2] - start[2])
    return np.array([xy[0], xy[1], wrap_angle(yaw)])


def _segment(trajectory, progress):
    index = int(np.clip(np.floor(progress), 0, last_progress(trajectory) - 1))
    return index, progress - index


def _segment_velocity(trajectory, index):
    start, end = trajectory.base_poses[index], trajectory.base_poses[index + 1]
    heading = np.array([np.cos(start[2]), np.sin(start[2])])
    linear = (end[:2] - start[:2]) @ heading / trajectory.sample_period
    angular = wrap_angle(end[2] - start[2]) / trajectory.sample_period
    return linear, angular


def reference_at(trajectory, progress):
    index, fraction = _segment(trajectory, progress)
    pose = _interpolate_pose(
        trajectory.base_poses[index], trajectory.base_poses[index + 1], fraction
    )
    angles = trajectory.door_angles
    door_angle = angles[index] + fraction * (angles[index + 1] - angles[index])
    arm_q = trajectory.arm_qs[index] + fraction * (
        trajectory.arm_qs[index + 1] - trajectory.arm_qs[index]
    )
    linear, angular = _segment_velocity(trajectory, index)
    return Reference(pose, door_angle, linear, angular, arm_q)


SWEEP_SLACK = 0.05


def piece_count(sweep, max_sweep):
    """Pieces needed to cover sweep, allowing each a small slack over max_sweep."""
    return max(1, int(np.ceil(sweep / max_sweep - SWEEP_SLACK)))


def split_by_sweep(trajectory, max_sweep):
    """Consecutive pieces sharing their boundary sample, each opening the door by the same amount, about max_sweep
    at most; the boundary is the first sample that reaches the piece's share."""
    angles = trajectory.door_angles
    count = piece_count(angles[-1] - angles[0], max_sweep)
    targets = angles[0] + (angles[-1] - angles[0]) * np.arange(1, count) / count
    inner = [int(np.searchsorted(angles, target - 1e-9)) for target in targets]
    bounds = sorted(set([0] + inner + [len(angles) - 1]))
    return [
        trajectory._replace(
            base_poses=trajectory.base_poses[start : end + 1],
            door_angles=angles[start : end + 1],
            arm_qs=trajectory.arm_qs[start : end + 1],
        )
        for start, end in zip(bounds[:-1], bounds[1:])
    ]


def from_angle(trajectory, angle):
    """The part of the trajectory from the last door angle at or below angle, at least its final two samples. Base
    motion planned at that door angle is kept."""
    angles = trajectory.door_angles
    last = max(int(np.searchsorted(angles, angle + 1e-9)) - 1, 0)
    first = int(np.searchsorted(angles, angles[last] - 1e-9))
    start = min(first, len(angles) - 2)
    return trajectory._replace(
        base_poses=trajectory.base_poses[start:],
        door_angles=angles[start:],
        arm_qs=trajectory.arm_qs[start:],
    )


def straight_drive(start_pose, distance, door_angle, arm_q, side, speed):
    """The base drives distance along its heading (backwards when negative), the door and the arm stay put."""
    count = max(2, int(np.ceil(abs(distance) / speed / SAMPLE_PERIOD)) + 1)
    heading = np.array([np.cos(start_pose[2]), np.sin(start_pose[2]), 0.0])
    base_poses = np.array(
        [
            start_pose + fraction * distance * heading
            for fraction in np.linspace(0.0, 1.0, count)
        ]
    )
    return DoorTrajectory(
        base_poses,
        np.full(count, door_angle),
        SAMPLE_PERIOD,
        side,
        np.tile(arm_q, (count, 1)),
    )
