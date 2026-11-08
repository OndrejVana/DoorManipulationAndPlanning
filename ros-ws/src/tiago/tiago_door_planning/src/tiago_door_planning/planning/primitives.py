import collections

import numpy as np

from ..model.geometry import wrap_angle

Primitive = collections.namedtuple("Primitive", "cell_offset heading_change samples length turn reverse")


def heading_angle(heading_bin, heading_bins):
    return 2.0 * np.pi * heading_bin / heading_bins


def _end_offset(start_heading, end_heading, cells):
    mean_heading = start_heading + wrap_angle(end_heading - start_heading) / 2.0
    return np.round(cells * np.array([np.cos(mean_heading), np.sin(mean_heading)])).astype(int)


def _hermite_points(start, end, start_tangent, end_tangent, fractions):
    points = []
    for t in fractions:
        h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
        h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
        points.append(h00 * start + h10 * start_tangent + h01 * end + h11 * end_tangent)
    return np.array(points)


def _curve_samples(start_heading, end_heading, end_xy, reverse, sample_count):
    fractions = np.linspace(0.0, 1.0, sample_count + 1)[1:]
    length = np.linalg.norm(end_xy)
    sign = -1.0 if reverse else 1.0
    start_tangent = sign * length * np.array([np.cos(start_heading), np.sin(start_heading)])
    end_tangent = sign * length * np.array([np.cos(end_heading), np.sin(end_heading)])
    points = _hermite_points(np.zeros(2), end_xy, start_tangent, end_tangent, fractions)
    headings = start_heading + fractions * wrap_angle(end_heading - start_heading)
    return np.column_stack([points, headings])


def _rotation_samples(start_heading, end_heading, sample_count):
    fractions = np.linspace(0.0, 1.0, sample_count + 1)[1:]
    headings = start_heading + fractions * wrap_angle(end_heading - start_heading)
    return np.column_stack([np.zeros(sample_count), np.zeros(sample_count), headings])


def make_primitive(heading_bin, heading_change, cells, heading_bins, resolution, sample_count):
    """Primitive from heading_bin; cells < 0 drives backwards, cells == 0 rotates in place."""
    start_heading = heading_angle(heading_bin, heading_bins)
    end_heading = heading_angle(heading_bin + heading_change, heading_bins)
    turn = abs(wrap_angle(end_heading - start_heading))
    if cells == 0:
        samples = _rotation_samples(start_heading, end_heading, sample_count)
        return Primitive((0, 0), heading_change, samples, 0.0, turn, False)
    offset = _end_offset(start_heading, end_heading, abs(cells)) * int(np.sign(cells))
    end_xy = offset * resolution
    samples = _curve_samples(start_heading, end_heading, end_xy, cells < 0, sample_count)
    samples[-1, :2] = end_xy
    return Primitive(tuple(offset), heading_change, samples, float(np.linalg.norm(end_xy)), turn, cells < 0)


def primitive_table(heading_bins, resolution, step_cells, sample_count):
    """Primitives for every start heading: forward/back straight and turning, plus in-place rotation."""
    motions = [
        (0, step_cells),
        (1, step_cells),
        (-1, step_cells),
        (0, -step_cells),
        (1, -step_cells),
        (-1, -step_cells),
        (1, 0),
        (-1, 0),
    ]
    return {
        heading_bin: [
            make_primitive(heading_bin, change, cells, heading_bins, resolution, sample_count)
            for change, cells in motions
        ]
        for heading_bin in range(heading_bins)
    }


def _turn_in_place(xy, start_yaw, end_yaw, step):
    change = wrap_angle(end_yaw - start_yaw)
    count = int(np.ceil(abs(change) / step - 1e-9))
    return [np.array([xy[0], xy[1], wrap_angle(start_yaw + change * f)]) for f in np.linspace(0.0, 1.0, count + 1)[1:]]


def turn_drive_turn(start, end, linear_step, angular_step):
    """Poses a differential drive passes from start to end, no more than a step apart: it turns toward end, drives
    straight there (backwards when that turns less) and turns to end's heading. Returns the poses and the turn."""
    offset = end[:2] - start[:2]
    distance = float(np.linalg.norm(offset))
    heading = start[2]
    if distance > 1e-9:
        heading = np.arctan2(offset[1], offset[0])
        if abs(wrap_angle(heading - start[2])) > np.pi / 2.0:
            heading = wrap_angle(heading + np.pi)
    count = int(np.ceil(distance / linear_step - 1e-9))
    drive = [np.array([*(start[:2] + offset * f), heading]) for f in np.linspace(0.0, 1.0, count + 1)[1:]]
    poses = _turn_in_place(start[:2], start[2], heading, angular_step) + drive
    poses += _turn_in_place(end[:2], heading, end[2], angular_step)
    turn = abs(wrap_angle(heading - start[2])) + abs(wrap_angle(end[2] - heading))
    return (poses or [np.array(end, dtype=float)]), distance, turn
