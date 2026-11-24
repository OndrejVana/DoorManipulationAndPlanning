"""Locking the door model: many tag-based samples of the door averaged into one, once they agree."""

import collections

import numpy as np

DoorSample = collections.namedtuple("DoorSample", "hinge handle hinge_side push tag_id")

Door = collections.namedtuple("Door", "hinge handle hinge_side push tag_id frame")

LockParams = collections.namedtuple("LockParams", "min_samples max_position_spread max_yaw_spread")


def mean_rotation(rotations):
    """The rotation nearest the element-wise mean of the rotations."""
    u, _, vt = np.linalg.svd(np.mean(rotations, axis=0))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def mean_pose(poses):
    pose = np.eye(4)
    pose[:3, :3] = mean_rotation([p[:3, :3] for p in poses])
    pose[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    return pose


def _yaw_spread(poses):
    """Spread of the handle normal's heading (z axis projected to the floor), circular std."""
    yaws = np.array([np.arctan2(p[1, 2], p[0, 2]) for p in poses])
    length = np.hypot(np.mean(np.cos(yaws)), np.mean(np.sin(yaws)))
    return np.sqrt(-2.0 * np.log(max(length, 1e-12)))


def _position_spread(poses):
    xy = np.array([p[:2, 3] for p in poses])
    return float(np.max(np.std(xy, axis=0)))


def _inliers(samples, max_spread):
    """The largest group of samples whose hinges and handles all lie within 3 max_spread of one sample's. A far tag
    can flip between two poses (the planar pose ambiguity), so the samples may form two groups with nothing at
    their median."""
    if not samples:
        return []
    hinges = np.array([s.hinge[:2, 3] for s in samples])
    handles = np.array([s.handle[:2, 3] for s in samples])

    def close(points):
        return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2) <= 3.0 * max_spread

    neighbours = close(hinges) & close(handles)
    centre = int(np.argmax(neighbours.sum(axis=1)))
    return [sample for sample, keep in zip(samples, neighbours[centre]) if keep]


def lock_door(samples, params, frame):
    """(Door, None) from the samples of the most recently seen tag, outliers left out, or (None, reason) when they
    are too few or disagree by more than the spreads allow."""
    if not samples:
        return None, "the door tag was not seen"
    tag_id = samples[-1].tag_id
    seen = [s for s in samples if s.tag_id == tag_id]
    same_tag = _inliers(seen, params.max_position_spread)
    if len(same_tag) < params.min_samples:
        return None, "only %d of %d samples of tag %d within the spread (need %d)" % (
            len(same_tag),
            len(seen),
            tag_id,
            params.min_samples,
        )
    if len({(s.hinge_side, s.push) for s in same_tag}) > 1:
        return None, "tag %d came with different hinge sides or interactions" % tag_id
    hinges = [s.hinge for s in same_tag]
    handles = [s.handle for s in same_tag]
    spread = max(_position_spread(hinges), _position_spread(handles))
    if spread > params.max_position_spread:
        return None, "door poses spread %.0f mm" % (1000.0 * spread)
    yaw_spread = _yaw_spread(handles)
    if yaw_spread > params.max_yaw_spread:
        return None, "door normal spread %.1f deg" % np.degrees(yaw_spread)
    first = same_tag[-1]
    door = Door(mean_pose(hinges), mean_pose(handles), first.hinge_side, first.push, tag_id, frame)
    return door, None
