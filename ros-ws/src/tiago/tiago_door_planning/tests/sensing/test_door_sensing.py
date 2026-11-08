import numpy as np
import pytest

from tiago_door_planning.model.door_scene import DoorDimensions, door_box, scene_from_door_poses, wall_boxes
from tiago_door_planning.sensing.door_sensing import (
    DoorReading,
    ScanParams,
    door_angle_at_least,
    door_angle_from_points,
    door_estimate,
    door_seen_ahead,
    hand_off_door,
    grasp_lost,
    ray_entries,
    scan_points,
)
from tiago_door_planning.model.geometry import Box, make_pose

DIMENSIONS = DoorDimensions(width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0)
PARAMS = ScanParams(0.03, 10, np.radians(1.0), np.radians(120.0), 0.15, 3)


def make_scene():
    return scene_from_door_poses(
        make_pose([0.0, 0.0, 0.0], 0.0), make_pose([0.75, 0.0, 1.0], 0.0), "left", False, DIMENSIONS
    )


def face_points(box, count, side):
    """Points along the long vertical face of a box at laser height."""
    along = np.linspace(-0.95, 0.95, count) * box.half_extents[0]
    local = np.column_stack([along, np.full(count, side * box.half_extents[1]), np.zeros(count)])
    return (local @ box.pose[:3, :3].T + box.pose[:3, 3])[:, :2]


@pytest.mark.parametrize("degrees", [0.0, 30.0, 55.0, 90.0])
def test_angle_found_from_the_leaf_beside_the_walls(degrees):
    scene = make_scene()
    leaf = face_points(door_box(scene, np.radians(degrees)), 60, 1.0)
    walls = np.vstack([face_points(box, 40, 1.0) for box in wall_boxes(scene)])

    angle = door_angle_from_points(scene, np.vstack([leaf, walls]), PARAMS)

    assert np.degrees(angle) == pytest.approx(degrees, abs=1.0)


def test_no_angle_without_points_on_the_leaf():
    scene = make_scene()
    far_away = np.array([[5.0, 5.0], [5.1, 5.0]])

    assert door_angle_from_points(scene, far_away, PARAMS) is None


def test_scan_points_skip_invalid_ranges():
    points = scan_points([1.0, np.inf, 2.0, 30.0], 0.0, np.pi / 2.0, 25.0)

    assert points == pytest.approx(np.array([[1.0, 0.0], [-2.0, 0.0]]), abs=1e-9)


def test_grasp_lost_only_when_the_door_lags_behind_the_hand():
    tolerance = np.radians(10.0)

    assert grasp_lost(np.radians(60.0), np.radians(45.0), tolerance)
    assert not grasp_lost(np.radians(50.0), np.radians(45.0), tolerance)
    assert not grasp_lost(np.radians(40.0), np.radians(60.0), tolerance)
    assert not grasp_lost(np.radians(60.0), None, tolerance)


def test_angles_below_the_known_door_angle_are_not_candidates():
    scene = make_scene()
    walls = np.vstack([face_points(box, 40, 1.0) for box in wall_boxes(scene)])
    closed_leaf = face_points(door_box(scene, 0.0), 60, 1.0)
    points = np.vstack([closed_leaf, walls])

    assert np.degrees(door_angle_from_points(scene, points, PARAMS)) == pytest.approx(0.0, abs=1.0)
    assert door_angle_from_points(scene, points, PARAMS, np.radians(60.0)) is None


def ring_walls():
    return [
        Box(make_pose(center, yaw), np.array([3.0, 0.05, 1.0]))
        for center, yaw in [
            ([0.0, 3.0, 1.0], 0.0),
            ([0.0, -3.0, 1.0], 0.0),
            ([3.0, 0.0, 1.0], np.pi / 2.0),
            ([-3.0, 0.0, 1.0], np.pi / 2.0),
        ]
    ]


def laser_hits(origin, boxes, count=720):
    """Hit points of rays from origin on the boxes and on whichever box each ray meets first."""
    directions = np.linspace(-np.pi, np.pi, count, endpoint=False)
    ends = origin + 5.0 * np.column_stack([np.cos(directions), np.sin(directions)])
    entries = np.min([ray_entries(origin, ends, box) for box in boxes], axis=0)
    first = [int(np.argmin([ray_entries(origin, end[None], box)[0] for box in boxes])) for end in ends]
    hit = np.isfinite(entries)
    return origin + entries[hit, None] * (ends[hit] - origin), np.array(first)[hit]


def unseen_leaf_scan(scene, origin, degrees):
    points, first = laser_hits(origin, [door_box(scene, np.radians(degrees))] + ring_walls())
    return points[first != 0]


@pytest.mark.parametrize("degrees", [20.0, 60.0, 85.0])
def test_rays_past_a_pushed_leaf_the_laser_cannot_see_bound_it_closely(degrees):
    scene = make_scene()
    origin = np.array([0.5, 0.8])
    unseen_leaf = unseen_leaf_scan(scene, origin, degrees)

    at_least = door_angle_at_least(scene, origin, unseen_leaf, PARAMS)

    assert door_angle_from_points(scene, unseen_leaf, PARAMS) is None
    assert degrees - 5.0 <= np.degrees(at_least) <= degrees + 0.5


@pytest.mark.parametrize("degrees", [20.0, 60.0, 85.0])
def test_the_bound_never_exceeds_a_pulled_leaf_the_laser_cannot_see(degrees):
    scene = make_scene()
    origin = np.array([0.5, -0.8])

    at_least = door_angle_at_least(scene, origin, unseen_leaf_scan(scene, origin, degrees), PARAMS)

    assert np.degrees(at_least) <= degrees + 0.5


def test_door_seen_ahead_of_the_hand_that_should_hold_it():
    tolerance = np.radians(10.0)

    assert door_seen_ahead(np.radians(70.0), DoorReading(None, np.radians(85.0)), tolerance)
    assert not door_seen_ahead(np.radians(70.0), DoorReading(None, np.radians(75.0)), tolerance)


def test_door_estimate_trusts_the_hand_unless_the_laser_contradicts_it():
    tolerance = np.radians(10.0)
    hand = np.radians(70.0)

    assert door_estimate(hand, None, tolerance) == hand
    assert door_estimate(hand, DoorReading(np.radians(72.0), np.radians(68.0)), tolerance) == hand
    assert door_estimate(hand, DoorReading(np.radians(74.0), np.radians(74.0)), tolerance) == hand
    assert door_estimate(hand, DoorReading(np.radians(50.0), np.radians(45.0)), tolerance) == np.radians(50.0)
    assert door_estimate(hand, DoorReading(np.radians(85.0), np.radians(85.0)), tolerance) == np.radians(85.0)
    assert door_estimate(hand, DoorReading(None, np.radians(86.0)), tolerance) == hand


def test_hand_is_off_the_door_when_the_laser_sees_the_door_elsewhere():
    tolerance = np.radians(10.0)
    hand = np.radians(70.0)

    assert not hand_off_door(hand, None, tolerance)
    assert not hand_off_door(hand, DoorReading(np.radians(72.0), np.radians(68.0)), tolerance)
    assert hand_off_door(hand, DoorReading(np.radians(50.0), np.radians(45.0)), tolerance)
    assert hand_off_door(hand, DoorReading(np.radians(85.0), np.radians(82.0)), tolerance)
    assert hand_off_door(hand, DoorReading(None, np.radians(86.0)), tolerance)


def test_a_scan_that_sees_through_every_angle_bounds_nothing():
    scene = make_scene()
    origin = np.array([0.5, 0.8])
    hinge_side_points = np.array(
        [[0.45, -2.0 + 0.02 * i] for i in range(40)] + [[-2.0 + 0.1 * i, -2.5] for i in range(40)]
    )
    sweep = np.array([[np.cos(a) * 2.5, -np.sin(a) * 2.5] for a in np.linspace(0.0, np.radians(120.0), 200)])

    at_least = door_angle_at_least(scene, origin, np.vstack([hinge_side_points, sweep]), PARAMS, np.radians(30.0))

    assert at_least == pytest.approx(np.radians(30.0))
