import collections

import numpy as np

from ..model.door_scene import door_box
from ..model.geometry import points_to_box_distances

ScanParams = collections.namedtuple(
    "ScanParams",
    "inlier_distance min_inliers angle_step max_angle hinge_clearance max_rays_through",
)


def scan_points(ranges, angle_min, angle_increment, range_max):
    """(N, 2) points in the laser frame from the valid ranges of a scan."""
    ranges = np.asarray(ranges, dtype=float)
    angles = angle_min + angle_increment * np.arange(len(ranges))
    valid = np.isfinite(ranges) & (ranges < range_max)
    return np.column_stack(
        [ranges[valid] * np.cos(angles[valid]), ranges[valid] * np.sin(angles[valid])]
    )


def _points_near_leaf(scene, points_xy, params):
    distances = np.linalg.norm(points_xy - scene.hinge[:2], axis=1)
    near = (distances >= params.hinge_clearance) & (
        distances <= scene.dimensions.width + params.inlier_distance
    )
    points = points_xy[near]
    return np.column_stack(
        [points, np.full(len(points), scene.dimensions.height / 2.0)]
    )


def leaf_distances(scene, points, angle, inlier_distance):
    """Distances of the points to the leaf surface, truncated at inlier_distance."""
    return np.minimum(
        np.abs(points_to_box_distances(points, door_box(scene, angle))), inlier_distance
    )


def _candidate_angles(params, min_angle):
    return np.arange(max(0.0, min_angle), params.max_angle + 1e-9, params.angle_step)


def door_angle_from_points(scene, points_xy, params, min_angle=0.0):
    """The door angle from min_angle up whose leaf lies closest to the points, or None when too few points lie on it.
    A door without a spring does not close on its own, so angles below where it was are not candidates: the frame
    and wall line up with a closed leaf and win when the leaf itself is hard to see."""
    points = _points_near_leaf(scene, np.asarray(points_xy, dtype=float), params)
    if len(points) == 0:
        return None
    candidates = _candidate_angles(params, min_angle)
    distances = [
        leaf_distances(scene, points, angle, params.inlier_distance)
        for angle in candidates
    ]
    best = int(np.argmin([np.sum(d) for d in distances]))
    if np.sum(distances[best] < params.inlier_distance) < params.min_inliers:
        return None
    return float(candidates[best])


def ray_entries(origin, ends, box):
    """Fraction of the way from origin to each end where the ray enters the box footprint, inf where it misses."""
    start = (origin - box.pose[:2, 3]) @ box.pose[:2, :2]
    direction = (ends - box.pose[:2, 3]) @ box.pose[:2, :2] - start
    direction = np.where(np.abs(direction) < 1e-12, 1e-12, direction)
    low = (-box.half_extents[:2] - start) / direction
    high = (box.half_extents[:2] - start) / direction
    enter = np.max(np.minimum(low, high), axis=1)
    leave = np.min(np.maximum(low, high), axis=1)
    return np.where((enter <= leave) & (leave >= 0.0), np.maximum(enter, 0.0), np.inf)


def rays_through_leaf(scene, origin, points_xy, angle, inlier_distance):
    """Rays that went on well past where the leaf at this angle would have stopped them."""
    lengths = np.linalg.norm(points_xy - origin, axis=1)
    entries = ray_entries(origin, points_xy, door_box(scene, angle))
    return int(np.sum(entries * lengths < lengths - inlier_distance))


def door_angle_at_least(scene, origin, points_xy, params, min_angle=0.0):
    """The smallest angle from min_angle up that the laser does not see through: a leaf the laser cannot see itself
    is at least this far open. A scan that sees through every angle does not match the scene and tells nothing.
    """
    points_xy = np.asarray(points_xy, dtype=float)
    candidates = _candidate_angles(params, min_angle)
    for angle in candidates:
        if (
            rays_through_leaf(scene, origin, points_xy, angle, params.inlier_distance)
            <= params.max_rays_through
        ):
            return float(angle)
    return float(candidates[0])


DoorReading = collections.namedtuple("DoorReading", "angle at_least")


def scan_door_reading(scan, frame_from_laser, scene, params, min_angle=0.0):
    """The fitted door angle (None without enough points on the leaf) and the angle the door is open at least,
    from a LaserScan-like message, with the scene in the frame frame_from_laser maps into.
    """
    points = scan_points(
        scan.ranges, scan.angle_min, scan.angle_increment, scan.range_max
    )
    points = points @ frame_from_laser[:2, :2].T + frame_from_laser[:2, 3]
    return DoorReading(
        door_angle_from_points(scene, points, params, min_angle),
        door_angle_at_least(scene, frame_from_laser[:2, 3], points, params, min_angle),
    )


def grasp_lost(hand_angle, door_angle, tolerance):
    """The hand runs ahead of the door it should be moving."""
    return door_angle is not None and hand_angle - door_angle > tolerance


def unseen_leaf_bound(reading):
    """The free-space bound of a leaf the laser does not see, else None: from some spots, such as behind a pulled
    door, rays pass by a leaf the fit sees and the bound reads far too high."""
    return reading.at_least if reading.angle is None else None


def door_seen_ahead(hand_angle, reading, tolerance):
    """The laser shows the door further open than the hand holding it: the hand lost it or closed on air."""
    seen = reading.angle if reading.angle is not None else reading.at_least
    return seen - hand_angle > tolerance


def hand_off_door(hand_angle, reading, tolerance):
    return reading is not None and (
        grasp_lost(hand_angle, reading.angle, tolerance)
        or door_seen_ahead(hand_angle, reading, tolerance)
    )


def door_estimate(hand_angle, reading, tolerance):
    """Where the hand holds the door, since the laser reads a turned leaf a few degrees off, unless the laser fit sees
    the leaf more than tolerance away. The free-space bound alone never moves it: from some spots it reads far high.
    """
    if (
        reading is None
        or reading.angle is None
        or abs(reading.angle - hand_angle) <= tolerance
    ):
        return hand_angle
    return reading.angle
