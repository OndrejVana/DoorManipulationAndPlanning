import collections
import numpy as np

from .geometry import Box, make_pose, rotation_z, wrap_angle

DoorDimensions = collections.namedtuple(
    "DoorDimensions", ["width", "thickness", "height", "wall_thickness", "wall_length"]
)

DoorScene = collections.namedtuple(
    "DoorScene", ["hinge", "closed_yaw", "opening_sign", "handle_closed", "dimensions"]
)


def opening_sign(push, hinge_side):
    if hinge_side not in ("left", "right"):
        raise ValueError("hinge_side must be 'left' or 'right', got %r" % hinge_side)
    hinge_left = hinge_side == "left"
    return 1.0 if push == hinge_left else -1.0


def scene_from_door_poses(hinge_pose, handle_pose, hinge_side, push, dimensions):
    hinge = np.array([hinge_pose[0, 3], hinge_pose[1, 3], 0.0])
    to_handle = handle_pose[:2, 3] - hinge[:2]
    closed_yaw = np.arctan2(to_handle[1], to_handle[0])
    return DoorScene(
        hinge,
        closed_yaw,
        opening_sign(push, hinge_side),
        handle_pose.copy(),
        dimensions,
    )


def door_yaw(scene, angle):
    return scene.closed_yaw + scene.opening_sign * angle


def handle_radius(scene):
    return np.linalg.norm(scene.handle_closed[:2, 3] - scene.hinge[:2])


def rotation_about_hinge(scene, angle):
    rotation = np.eye(4)
    rotation[:3, :3] = rotation_z(scene.opening_sign * angle)
    rotation[:3, 3] = scene.hinge - rotation[:3, :3] @ scene.hinge
    return rotation


def handle_pose(scene, angle):
    return rotation_about_hinge(scene, angle) @ scene.handle_closed


def _door_axes(scene, angle):
    yaw = door_yaw(scene, angle)
    along = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    away_from_swing = rotation_z(-scene.opening_sign * np.pi / 2.0) @ along
    return yaw, along, away_from_swing


def door_box(scene, angle):
    dims = scene.dimensions
    yaw, along, away = _door_axes(scene, angle)
    center = scene.hinge + along * dims.width / 2.0 + away * dims.thickness / 2.0
    center[2] = dims.height / 2.0
    return Box(
        make_pose(center, yaw),
        np.array([dims.width, dims.thickness, dims.height]) / 2.0,
    )


def _wall_box(scene, start_offset, end_offset):
    dims = scene.dimensions
    _, along, away = _door_axes(scene, 0.0)
    middle = (start_offset + end_offset) / 2.0
    center = scene.hinge + along * middle + away * dims.wall_thickness / 2.0
    center[2] = dims.height / 2.0
    length = abs(end_offset - start_offset)
    return Box(
        make_pose(center, scene.closed_yaw),
        np.array([length, dims.wall_thickness, dims.height]) / 2.0,
    )


def wall_boxes(scene):
    dims = scene.dimensions
    hinge_side = _wall_box(scene, -dims.wall_length, 0.0)
    latch_side = _wall_box(scene, dims.width, dims.width + dims.wall_length)
    return [hinge_side, latch_side]


def obstacle_boxes(scene, angle):
    return wall_boxes(scene) + [door_box(scene, angle)]


def _hinge_frame_axes(scene):
    along = np.array([np.cos(scene.closed_yaw), np.sin(scene.closed_yaw)])
    swing = rotation_z(scene.opening_sign * np.pi / 2.0)[:2, :2] @ along
    return along, swing


def from_hinge_frame(scene, pose2d):
    along, swing = _hinge_frame_axes(scene)
    xy = scene.hinge[:2] + pose2d[0] * along + pose2d[1] * swing
    yaw = wrap_angle(scene.closed_yaw + scene.opening_sign * pose2d[2])
    return np.array([xy[0], xy[1], yaw])


def to_hinge_frame(scene, pose2d):
    along, swing = _hinge_frame_axes(scene)
    offset = np.asarray(pose2d[:2]) - scene.hinge[:2]
    yaw = wrap_angle(scene.opening_sign * (pose2d[2] - scene.closed_yaw))
    return np.array([offset @ along, offset @ swing, yaw])
