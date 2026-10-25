import collections
import numpy as np

from .door_scene import from_hinge_frame
from .geometry import wrap_angle

DoorTrajectory = collections.namedtuple(
    "DoorTrajectory",
    ["base_poses", "door_angles", "sample_period", "side", "elbow_hint"],
)

Reference = collections.namedtuple(
    "Reference", ["base_pose", "door_angle", "linear", "angular"]
)


def mirrored_side(side):
    return "right" if side == "left" else "left"


def trajectory_to_world(trajectory, scene):
    """Map a hinge-frame trajectory onto a scene; the arm swaps for doors opening the other way."""
    base_poses = np.array(
        [from_hinge_frame(scene, pose) for pose in trajectory.base_poses]
    )
    side = trajectory.side if scene.opening_sign > 0 else mirrored_side(trajectory.side)
    return trajectory._replace(base_poses=base_poses, side=side)


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
    linear, angular = _segment_velocity(trajectory, index)
    return Reference(pose, door_angle, linear, angular)


def save_trajectory(path, trajectory):
    np.savez(
        path,
        base_poses=trajectory.base_poses,
        door_angles=trajectory.door_angles,
        sample_period=trajectory.sample_period,
        side=trajectory.side,
        elbow_hint=trajectory.elbow_hint,
    )


def load_trajectory(path):
    data = np.load(path)
    return DoorTrajectory(
        data["base_poses"],
        data["door_angles"],
        float(data["sample_period"]),
        str(data["side"]),
        data["elbow_hint"],
    )
