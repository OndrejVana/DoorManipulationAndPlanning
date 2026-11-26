import collections

import numpy as np

from ..model.door_scene import door_yaw

ComplianceParams = collections.namedtuple(
    "ComplianceParams", "deadband compliance max_offset offset_leak_s vertical"
)

RADIAL, TANGENTIAL, VERTICAL = 0, 1, 2


def door_axes(scene, angle):
    """Rows: along the leaf away from the hinge, the way the handle moves as the door opens, up."""
    yaw = door_yaw(scene, angle)
    radial = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.array([radial, scene.opening_sign * np.cross(up, radial), up])


def deadband(values, width):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.maximum(np.abs(values) - width, 0.0)


def admittance_step(offset, door_force, dt, params):
    """Offset (radial, tangential, vertical) after dt: it moves with the force beyond the deadband, leaks back to zero
    and is clipped to max_offset. It never reaches along the door's motion, and up or down only with params.vertical:
    a pull hook the bar stops vertically never relaxes the force and the offset only slides it off."""
    rate = params.compliance * deadband(door_force, params.deadband)
    rate -= np.asarray(offset) / params.offset_leak_s
    rate[TANGENTIAL] = 0.0
    if not params.vertical:
        rate[VERTICAL] = 0.0
    return np.clip(offset + rate * dt, -params.max_offset, params.max_offset)


def offset_in_world(offset, axes):
    return np.asarray(offset) @ axes
