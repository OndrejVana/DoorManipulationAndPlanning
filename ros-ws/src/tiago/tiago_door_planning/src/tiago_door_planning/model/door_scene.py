import collections
import numpy as np

from .geometry import Box, make_pose, rotation_z, wrap_angle

DoorDimensions = collections.namedtuple(
    "DoorDimensions", "width thickness height wall_thickness wall_length"
)

DoorScene = collections.namedtuple(
    "DoorScene", "hinge closed_yaw opening_sign handle_closed dimensions"
)


def opening_sign(push, hinge_side):
    if hinge_side not in ("left", "right"):
        raise ValueError("hinge_side must be 'left' or 'right', got %r" % hinge_side)
    hinge_left = hinge_side == "left"
    return 1.0 if push == hinge_left else -1.0


def _along_door_plane(to_handle, handle_pose):
    """Hinge-to-handle direction without the handle's standoff from the door; the handle z axis is the door normal."""
    normal = handle_pose[:2, 2]
    if np.linalg.norm(normal) < 0.5:
        return to_handle
    normal = normal / np.linalg.norm(normal)
    return to_handle - (to_handle @ normal) * normal


def scene_from_door_poses(hinge_pose, handle_pose, hinge_side, push, dimensions):
    hinge = np.array([hinge_pose[0, 3], hinge_pose[1, 3], 0.0])
    along = _along_door_plane(handle_pose[:2, 3] - hinge[:2], handle_pose)
    closed_yaw = np.arctan2(along[1], along[0])
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


def door_angle_at(scene, point_xy):
    """Door angle at which the handle would lie in the direction of point_xy, seen from the hinge."""
    to_point = np.asarray(point_xy) - scene.hinge[:2]
    to_handle = scene.handle_closed[:2, 3] - scene.hinge[:2]
    return scene.opening_sign * wrap_angle(
        np.arctan2(to_point[1], to_point[0]) - np.arctan2(to_handle[1], to_handle[0])
    )


def rotation_about_hinge(scene, angle):
    rotation = np.eye(4)
    rotation[:3, :3] = rotation_z(scene.opening_sign * angle)
    rotation[:3, 3] = scene.hinge - rotation[:3, :3] @ scene.hinge
    return rotation


def handle_pose(scene, angle):
    return rotation_about_hinge(scene, angle) @ scene.handle_closed


def back_handle(scene):
    """The handle on the other face: mirrored through the middle of the leaf, z out of the other face and, pushed on,
    y down like a push handle whichever way up the tag model turned the front one."""
    _, _, away = _door_axes(scene, 0.0)
    middle = scene.hinge + away * scene.dimensions.thickness / 2.0
    handle = scene.handle_closed.copy()
    handle[:3, 3] -= 2.0 * ((handle[:3, 3] - middle) @ away) * away
    out = -handle[:3, 2] * [1.0, 1.0, 0.0]
    out /= np.linalg.norm(out)
    down = np.array([0.0, 0.0, -1.0])
    handle[:3, :3] = np.column_stack([np.cross(down, out), down, out])
    return handle


def back_scene(scene):
    return scene._replace(handle_closed=back_handle(scene))


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


def swept_polygon(scene, min_angle, max_angle, margin, arc_points=12):
    """(N, 2) polygon around everywhere the leaf can be between min_angle and max_angle, grown by margin: the swept
    sector plus the leaf's thickness at min_angle on the side it does not swing to."""
    dims = scene.dimensions
    _, along, away = _door_axes(scene, min_angle)
    radius = dims.width + margin
    behind = away * (dims.thickness + margin)
    hinge = scene.hinge
    corners = [hinge - along * margin + behind, hinge + along * radius + behind]
    arc = [
        hinge + _door_axes(scene, angle)[1] * radius
        for angle in np.linspace(min_angle, max_angle, arc_points)
    ]
    _, last_along, last_away = _door_axes(scene, max_angle)
    closing = [
        hinge + last_along * radius - last_away * margin,
        hinge - last_away * margin,
        hinge - last_along * margin - last_away * margin,
    ]
    return np.array([point[:2] for point in corners + arc + closing])
