import collections
import numpy as np
from scipy.ndimage import distance_transform_edt

from .geometry import Sphere, points_to_boxes_distances, pose2d_to_matrix

CollisionParams = collections.namedtuple(
    "CollisionParams", "base_radius base_center_height margin"
)

DistanceGrid = collections.namedtuple("DistanceGrid", "distance resolution origin")

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


def spheres_in_world(base_pose, spheres):
    base = pose2d_to_matrix(base_pose)
    centers = (
        np.array([s.center for s in spheres]).reshape(-1, 3) @ base[:3, :3].T
        + base[:3, 3]
    )
    return [
        Sphere(center, s.radius, s.is_gripper) for center, s in zip(centers, spheres)
    ]


def robot_clearance(walls, door, grid, base_pose, world_spheres, mode, params):
    """The base and every arm sphere against all boxes in one go; in GRASP mode the gripper may touch the door."""
    centers = np.array(
        [[base_pose[0], base_pose[1], params.base_center_height]]
        + [s.center for s in world_spheres]
    )
    radii = np.array([params.base_radius] + [s.radius for s in world_spheres])
    clearances = points_to_boxes_distances(centers, walls + [door]) - radii[:, None]
    if mode == GRASP:
        grippers = np.array([False] + [s.is_gripper for s in world_spheres])
        clearances[grippers, -1] = np.inf
    clearance = float(np.min(clearances))
    if grid is not None:
        clearance = min(clearance, base_costmap_clearance(grid, base_pose, params))
    return clearance


def spheres_clearance(spheres, others):
    if not spheres or not others:
        return np.inf
    centers = np.array([s.center for s in spheres])
    other_centers = np.array([s.center for s in others])
    gaps = np.linalg.norm(centers[:, None, :] - other_centers[None, :, :], axis=2)
    radii = (
        np.array([s.radius for s in spheres])[:, None]
        + np.array([s.radius for s in others])[None, :]
    )
    return float(np.min(gaps - radii))


def arms_collide(active_spheres, passive_spheres, params):
    return spheres_clearance(active_spheres, passive_spheres) < params.margin


def robot_collides(walls, door, grid, base_pose, world_spheres, mode, params):
    return (
        robot_clearance(walls, door, grid, base_pose, world_spheres, mode, params)
        < params.margin
    )
