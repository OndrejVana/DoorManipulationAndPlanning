import collections
import numpy as np
from scipy.ndimage import distance_transform_edt

from .geometry import (
    Sphere,
    points_to_box_distances,
    pose2d_to_matrix,
    sphere_to_box_distance,
)

CollisionParams = collections.namedtuple(
    "CollisionParams", ["base_radius", "base_center_height", "margin"]
)

DistanceGrid = collections.namedtuple(
    "DistanceGrid", ["distance", "resolution", "origin"]
)

GRASP = "grasp"
FREE = "free"


def distance_grid(occupancy, resolution, origin, occupied_threshold):
    """Precompute metric distance to the nearest occupied cell; occupancy is a (height, width) array."""
    free = np.asarray(occupancy) < occupied_threshold
    return DistanceGrid(
        distance_transform_edt(free) * resolution,
        resolution,
        np.asarray(origin, dtype=float),
    )


def grid_distance_at(grid, xy):
    column, row = ((np.asarray(xy) - grid.origin) / grid.resolution).astype(int)
    height, width = grid.distance.shape
    if not (0 <= row < height and 0 <= column < width):
        return np.inf
    return grid.distance[row, column]


def base_costmap_clearance(grid, base_pose, params):
    return grid_distance_at(grid, base_pose[:2]) - params.base_radius


def base_scene_clearance(walls, door, base_pose, params):
    center = np.array([base_pose[0], base_pose[1], params.base_center_height])
    return min(
        sphere_to_box_distance(center, params.base_radius, box)
        for box in walls + [door]
    )


def spheres_in_world(base_pose, spheres):
    base = pose2d_to_matrix(base_pose)
    centers = (
        np.array([s.center for s in spheres]).reshape(-1, 3) @ base[:3, :3].T
        + base[:3, 3]
    )
    return [
        Sphere(center, s.radius, s.is_gripper) for center, s in zip(centers, spheres)
    ]


def _min_clearance(centers, radii, box):
    if len(centers) == 0:
        return np.inf
    return float(np.min(points_to_box_distances(centers, box) - radii))


def arm_scene_clearance(walls, door, world_spheres, mode):
    if not world_spheres:
        return np.inf
    centers = np.array([s.center for s in world_spheres])
    radii = np.array([s.radius for s in world_spheres])
    checks_door = np.array(
        [not (s.is_gripper and mode == GRASP) for s in world_spheres]
    )
    wall_clearance = min(_min_clearance(centers, radii, box) for box in walls)
    door_clearance = _min_clearance(centers[checks_door], radii[checks_door], door)
    return min(wall_clearance, door_clearance)


def robot_clearance(walls, door, grid, base_pose, world_spheres, mode, params):
    clearances = [
        base_scene_clearance(walls, door, base_pose, params),
        arm_scene_clearance(walls, door, world_spheres, mode),
    ]
    if grid is not None:
        clearances.append(base_costmap_clearance(grid, base_pose, params))
    return min(clearances)


def robot_collides(walls, door, grid, base_pose, world_spheres, mode, params):
    return (
        robot_clearance(walls, door, grid, base_pose, world_spheres, mode, params)
        < params.margin
    )
