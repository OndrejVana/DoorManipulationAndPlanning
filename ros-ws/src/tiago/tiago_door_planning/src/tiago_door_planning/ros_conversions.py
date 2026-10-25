import numpy as np
from .geometry import quaternion_to_rotation


def _matrix(translation, rotation):
    pose = np.eye(4)
    pose[:3, :3] = quaternion_to_rotation(
        rotation.x, rotation.y, rotation.z, rotation.w
    )
    pose[:3, 3] = [translation.x, translation.y, translation.z]
    return pose


def pose_msg_to_matrix(pose_msg):
    return _matrix(pose_msg.position, pose_msg.orientation)


def transform_msg_to_matrix(transform_msg):
    return _matrix(transform_msg.translation, transform_msg.rotation)


def joint_positions(joint_state_msg, names):
    index = {name: i for i, name in enumerate(joint_state_msg.name)}
    return np.array([joint_state_msg.position[index[name]] for name in names])


def joint_position_map(joint_state_msg):
    return dict(zip(joint_state_msg.name, joint_state_msg.position))
