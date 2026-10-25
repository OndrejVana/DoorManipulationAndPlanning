import collections
import numpy as np

Box = collections.namedtuple("Box", ["pose", "half_extents"])
Sphere = collections.namedtuple("Sphere", ["center", "radius", "is_gripper"])


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotation_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quaternion_to_rotation(x, y, z, w):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def make_pose(position, yaw):
    pose = np.eye(4)
    pose[:3, :3] = rotation_z(yaw)
    pose[:3, 3] = position
    return pose


def pose2d_to_matrix(pose2d):
    x, y, yaw = pose2d
    return make_pose([x, y, 0.0], yaw)


def matrix_to_pose2d(pose):
    yaw = np.arctan2(pose[1, 0], pose[0, 0])
    return np.array([pose[0, 3], pose[1, 3], yaw])


def invert(pose):
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def transform_point(pose, point):
    return pose[:3, :3] @ np.asarray(point, dtype=float) + pose[:3, 3]


def rotation_log(rotation):
    angle = np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    axis_terms = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-9:
        return axis_terms / 2.0
    return axis_terms * angle / (2.0 * np.sin(angle))


def local_pose_error(current, target):
    """Translation and rotation from current to target, both in the current frame."""
    rotation = current[:3, :3].T
    translation = rotation @ (target[:3, 3] - current[:3, 3])
    return np.concatenate([translation, rotation_log(rotation @ target[:3, :3])])


def damped_pseudo_inverse(jacobian, damping):
    regularized = jacobian @ jacobian.T + damping**2 * np.eye(jacobian.shape[0])
    return jacobian.T @ np.linalg.inv(regularized)


def points_to_box_distances(points, box):
    """Signed distances from an (N, 3) array of points to a box."""
    local = (np.asarray(points, dtype=float) - box.pose[:3, 3]) @ box.pose[:3, :3]
    excess = np.abs(local) - box.half_extents
    outside = np.linalg.norm(np.maximum(excess, 0.0), axis=1)
    inside = np.minimum(np.max(excess, axis=1), 0.0)
    return outside + inside


def point_to_box_distance(point, box):
    return points_to_box_distances([point], box)[0]


def sphere_to_box_distance(center, radius, box):
    return point_to_box_distance(center, box) - radius
