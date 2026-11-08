import collections
import heapq

import numpy as np
from scipy.ndimage import label

from ..model.collision import (
    GRASP,
    base_costmap_clearance,
    robot_collides,
    spheres_in_world,
)
from ..model.door_scene import door_box, handle_pose, wall_boxes
from ..model.geometry import points_to_boxes_distances, wrap_angle
from .parallel import ForkedMap

FreeSpace = collections.namedtuple("FreeSpace", "connected origin cell_size")

GraspSpot = collections.namedtuple("GraspSpot", "pose grip q")

DriveMap = collections.namedtuple("DriveMap", "distance origin cell_size")

NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

MAX_GRASP_SPOTS = 5
BASE_ERROR = 0.03
CANDIDATES_PER_WORKER = 4


def _ring_offsets(params):
    steps = np.arange(-params.reach_max, params.reach_max + 1e-9, params.cell_size)
    offsets = np.array([(dx, dy) for dx in steps for dy in steps])
    distances = np.linalg.norm(offsets, axis=1)
    in_ring = (distances >= params.reach_min) & (distances <= params.reach_max)
    middle = (params.reach_min + params.reach_max) / 2.0
    order = np.argsort(np.abs(distances[in_ring] - middle))
    return offsets[in_ring][order]


def candidate_base_poses(handle_xy, params):
    """Base poses around the handle that face it within max_bearing, nearest the middle of the reach ring first."""
    headings = 2.0 * np.pi * np.arange(params.heading_bins) / params.heading_bins
    for offset in _ring_offsets(params):
        bearing_to_handle = np.arctan2(-offset[1], -offset[0])
        for yaw in headings:
            if abs(wrap_angle(bearing_to_handle - yaw)) <= params.max_bearing:
                yield np.array(
                    [handle_xy[0] + offset[0], handle_xy[1] + offset[1], yaw]
                )


def _bases_clear(problem, walls, door, xs, ys):
    """Whether the bare base is collision free at every (x, y) of the grid; rows are y."""
    params = problem.collision
    gx, gy = np.meshgrid(xs, ys)
    centers = np.column_stack(
        [gx.ravel(), gy.ravel(), np.full(gx.size, params.base_center_height)]
    )
    clearance = np.min(
        points_to_boxes_distances(centers, walls + [door]) - params.base_radius, axis=1
    )
    if problem.grid is not None:
        costmap = [
            base_costmap_clearance(problem.grid, center, params) for center in centers
        ]
        clearance = np.minimum(clearance, costmap)
    return (clearance >= params.margin).reshape(gx.shape)


def _cell_index(free_space, xy):
    column, row = np.round(
        (np.asarray(xy) - free_space.origin) / free_space.cell_size
    ).astype(int)
    return row, column


def _nearest_free_cell(free, cell):
    free_cells = np.argwhere(free)
    if len(free_cells) == 0:
        return cell
    return tuple(
        free_cells[np.argmin(np.linalg.norm(free_cells - np.asarray(cell), axis=1))]
    )


def connected_free_space(problem, walls, door, handle_xy, params):
    """Base positions that are collision free and connected to the start position."""
    start_xy = problem.start_pose[:2]
    margin = params.reach_max + params.cell_size
    origin = np.minimum(handle_xy, start_xy) - margin
    xs = np.arange(origin[0], max(handle_xy[0], start_xy[0]) + margin, params.cell_size)
    ys = np.arange(origin[1], max(handle_xy[1], start_xy[1]) + margin, params.cell_size)
    free = _bases_clear(problem, walls, door, xs, ys)
    labels, _ = label(free)
    free_space = FreeSpace(None, origin, params.cell_size)
    start_label = labels[_nearest_free_cell(free, _cell_index(free_space, start_xy))]
    return free_space._replace(connected=(labels == start_label) & free)


def is_connected(free_space, xy):
    row, column = _cell_index(free_space, xy)
    height, width = free_space.connected.shape
    return (
        0 <= row < height
        and 0 <= column < width
        and bool(free_space.connected[row, column])
    )


def _arm_clear(problem, checks, walls, door, pose, q):
    spheres = spheres_in_world(pose, checks.spheres(q))
    return not robot_collides(
        walls, door, problem.grid, pose, spheres, GRASP, problem.collision
    )


def _arm_grasp(problem, checks, walls, door, pose, angle):
    for seed in (problem.start_q, None):
        q = checks.solve(pose, angle, seed)
        if q is not None and _arm_clear(problem, checks, walls, door, pose, q):
            return q
    return None


def _grasp_at(problem, walls, door, pose, angle):
    for grip, checks in problem.grips.items():
        q = _arm_grasp(problem, checks, walls, door, pose, angle)
        if q is not None:
            return GraspSpot(pose, grip, q)
    return None


def _first_found(check, candidates, count, workers):
    """The first count results of check that are not None, in the order of the candidates."""
    checks = ForkedMap(check, workers)
    batch = CANDIDATES_PER_WORKER * workers
    found = []
    try:
        for first in range(0, len(candidates), batch):
            found += [
                result
                for result in checks(candidates[first : first + batch])
                if result is not None
            ]
            if len(found) >= count:
                break
    finally:
        checks.close()
    return found[:count]


def _connected_candidates(problem, walls, door, angle, params):
    handle_xy = handle_pose(problem.scene, angle)[:2, 3]
    free_space = connected_free_space(problem, walls, door, handle_xy, params)
    return [
        pose
        for pose in candidate_base_poses(handle_xy, params)
        if is_connected(free_space, pose[:2])
    ]


def _grasp_pose_exists_at(problem, angle, params):
    walls, door = wall_boxes(problem.scene), door_box(problem.scene, angle)
    candidates = _connected_candidates(problem, walls, door, angle, params)
    return bool(
        _first_found(
            lambda pose: _grasp_at(problem, walls, door, pose, angle),
            candidates,
            1,
            params.workers,
        )
    )


def _coarse(params):
    return params._replace(
        cell_size=2.0 * params.cell_size, heading_bins=params.heading_bins // 2
    )


def grasp_pose_exists(problem, angle, params):
    """Tries a coarse subset of the candidate poses first; only a miss pays for the full sampling."""
    return _grasp_pose_exists_at(
        problem, angle, _coarse(params)
    ) or _grasp_pose_exists_at(problem, angle, params)


def _shifted_poses(pose, distance):
    """The pose moved by distance forward, backward and to either side."""
    heading = np.array([np.cos(pose[2]), np.sin(pose[2])])
    side = np.array([-heading[1], heading[0]])
    return [
        np.array([*(pose[:2] + offset), pose[2]])
        for offset in (
            distance * heading,
            -distance * heading,
            distance * side,
            -distance * side,
        )
    ]


def _tolerates_base_error(problem, walls, door, spot, angle):
    """A released drive ends a few centimetres off; the grasp must not depend on hitting the spot exactly."""
    checks = problem.grips[spot.grip]
    for pose in _shifted_poses(spot.pose, BASE_ERROR):
        q = checks.solve(pose, angle, spot.q)
        if q is None or not _arm_clear(problem, checks, walls, door, pose, q):
            return False
    return True


def _regrasp_at(problem, walls, door, pose, angle, next_angle):
    """The grasp at angle, solved from a grasp at next_angle, so that the grip can also move the door on."""
    ahead = _grasp_at(
        problem, walls, door_box(problem.scene, next_angle), pose, next_angle
    )
    if ahead is None:
        return None
    checks = problem.grips[ahead.grip]
    q = checks.solve(pose, angle, ahead.q)
    if q is None or not _arm_clear(problem, checks, walls, door, pose, q):
        return None
    spot = GraspSpot(pose, ahead.grip, q)
    if not _tolerates_base_error(
        problem, walls, door, spot, angle
    ) or not checks.reachable(pose, angle, q):
        return None
    return spot


def grasp_spots(problem, angle, next_angle, params):
    """GraspSpots on the candidate grid from which some grip grasps the handle at angle and can move it on to
    next_angle, at most MAX_GRASP_SPOTS of them, nearest the robot first."""
    walls, door = wall_boxes(problem.scene), door_box(problem.scene, angle)
    candidates = _connected_candidates(problem, walls, door, angle, params)
    candidates.sort(key=lambda pose: np.linalg.norm(pose[:2] - problem.start_pose[:2]))
    return _first_found(
        lambda pose: _regrasp_at(problem, walls, door, pose, angle, next_angle),
        candidates,
        MAX_GRASP_SPOTS,
        params.workers,
    )


def _cell_inside(shape, cell):
    return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]


def _distances_from(free, sources, cell_size):
    """Dijkstra over 8-connected free cells."""
    distance = np.full(free.shape, np.inf)
    heap = []
    for cell in sources:
        if _cell_inside(free.shape, cell):
            distance[cell] = 0.0
            heapq.heappush(heap, (0.0, cell))
    while heap:
        current, cell = heapq.heappop(heap)
        if current > distance[cell]:
            continue
        for dr, dc in NEIGHBOURS:
            neighbour = (cell[0] + dr, cell[1] + dc)
            step = cell_size * np.hypot(dr, dc)
            if (
                _cell_inside(free.shape, neighbour)
                and free[neighbour]
                and current + step < distance[neighbour]
            ):
                distance[neighbour] = current + step
                heapq.heappush(heap, (current + step, neighbour))
    return distance


def drive_map(problem, angle, spots, params):
    """Driving distance from each free base cell to the nearest spot, for steering the drive around the leaf."""
    walls, door = wall_boxes(problem.scene), door_box(problem.scene, angle)
    handle_xy = handle_pose(problem.scene, angle)[:2, 3]
    free_space = connected_free_space(problem, walls, door, handle_xy, params)
    sources = [_cell_index(free_space, spot.pose[:2]) for spot in spots]
    distance = _distances_from(free_space.connected, sources, params.cell_size)
    return DriveMap(distance, free_space.origin, params.cell_size)


def drive_distance(drive, xy):
    """Distance to drive to the nearest spot; 0 where the map knows nothing, which keeps it a lower bound."""
    column, row = np.round((np.asarray(xy) - drive.origin) / drive.cell_size).astype(
        int
    )
    if not _cell_inside(drive.distance.shape, (row, column)) or not np.isfinite(
        drive.distance[row, column]
    ):
        return 0.0
    return float(drive.distance[row, column])
