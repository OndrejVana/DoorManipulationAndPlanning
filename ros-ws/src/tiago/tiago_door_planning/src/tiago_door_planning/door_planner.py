"""A* door-opening planner over (x, y, heading, door angle, grasped).

The lattice is anchored at the robot's start pose, so the start is exactly a lattice state. Arm feasibility is
injected through ArmChecks, which keeps this module free of kinematics libraries.
"""
import collections

import numpy as np

from .astar import astar
from .collision import GRASP, FREE, robot_collides, spheres_in_world
from .door_scene import door_box, wall_boxes
from .geometry import matrix_to_pose2d, pose2d_to_matrix
from .primitives import primitive_table

PlannerParams = collections.namedtuple("PlannerParams", [
    "resolution", "heading_bins", "step_cells", "samples_per_edge",
    "angle_step", "max_door_steps", "goal_angle",
    "linear_speed", "angular_speed", "door_rate", "reverse_penalty", "regrasp_time",
    "max_expansions",
])

ArmChecks = collections.namedtuple("ArmChecks", ["solve", "spheres"])

PlanningProblem = collections.namedtuple("PlanningProblem", [
    "scene", "grid", "start_pose", "start_angle", "start_q", "free_q", "arm", "collision",
])

State = collections.namedtuple("State", ["ix", "iy", "heading", "door", "grasped"])

EdgeData = collections.namedtuple("EdgeData", ["q", "poses", "angles", "duration"])

PathStep = collections.namedtuple("PathStep", ["poses", "angles", "duration", "grasped", "q"])


class Planner(object):
    def __init__(self, problem, params):
        self.problem = problem
        self.params = params
        self.primitives = primitive_table(params.heading_bins, params.resolution, params.step_cells,
                                          params.samples_per_edge)
        self.goal_door = int(np.ceil((params.goal_angle - problem.start_angle) / params.angle_step - 1e-9))
        self.walls = wall_boxes(problem.scene)
        self.door_boxes = {}

    def door_box_at(self, angle):
        key = round(angle, 6)
        if key not in self.door_boxes:
            self.door_boxes[key] = door_box(self.problem.scene, angle)
        return self.door_boxes[key]

    def door_angle(self, door_index):
        return self.problem.start_angle + door_index * self.params.angle_step

    def lattice_to_world(self, lattice_pose):
        return matrix_to_pose2d(pose2d_to_matrix(self.problem.start_pose) @ pose2d_to_matrix(lattice_pose))

    def state_lattice_pose(self, state):
        heading = 2.0 * np.pi * state.heading / self.params.heading_bins
        return np.array([state.ix * self.params.resolution, state.iy * self.params.resolution, heading])

    def edge_world_poses(self, state, primitive):
        origin = self.state_lattice_pose(state)
        lattice = primitive.samples + np.array([origin[0], origin[1], 0.0])
        return [self.lattice_to_world(pose) for pose in lattice]

    def is_clear(self, base_pose, door_angle, q, grasped):
        spheres = spheres_in_world(base_pose, self.problem.arm.spheres(q))
        mode = GRASP if grasped else FREE
        return not robot_collides(self.walls, self.door_box_at(door_angle), self.problem.grid, base_pose, spheres,
                                  mode, self.problem.collision)

    def edge_duration(self, primitive, door_steps):
        params = self.params
        times = [door_steps * params.angle_step / params.door_rate]
        if primitive is not None:
            times += [primitive.length / params.linear_speed, primitive.turn / params.angular_speed]
        return max(times)

    def edge_cost(self, primitive, door_steps):
        reverse = primitive is not None and primitive.reverse
        return self.edge_duration(primitive, door_steps) + (self.params.reverse_penalty if reverse else 0.0)

    def heuristic(self, state):
        remaining = max(0, self.goal_door - state.door)
        return remaining * self.params.angle_step / self.params.door_rate

    def is_goal(self, state):
        return state.door >= self.goal_door

    def _track_grasped(self, poses, angles, q):
        for pose, angle in zip(poses, angles):
            q = self.problem.arm.solve(pose, angle, q)
            if q is None or not self.is_clear(pose, angle, q, True):
                return None
        return q

    def _grasped_edge(self, state, q, primitive, door_steps):
        start_pose = self.lattice_to_world(self.state_lattice_pose(state))
        poses = self.edge_world_poses(state, primitive) if primitive else [start_pose] * self.params.samples_per_edge
        fractions = np.linspace(0.0, 1.0, len(poses) + 1)[1:]
        angles = [self.door_angle(state.door + door_steps * f) for f in fractions]
        q_end = self._track_grasped(poses, angles, q)
        if q_end is None:
            return None
        next_state = self.next_state(state, primitive, door_steps, True)
        data = EdgeData(q_end, poses, angles, self.edge_duration(primitive, door_steps))
        return next_state, self.edge_cost(primitive, door_steps), data

    def _free_edge(self, state, last_grasp_q, primitive):
        poses = self.edge_world_poses(state, primitive)
        angle = self.door_angle(state.door)
        if not all(self.is_clear(pose, angle, self.problem.free_q, False) for pose in poses):
            return None
        data = EdgeData(last_grasp_q, poses, [angle] * len(poses), self.edge_duration(primitive, 0))
        return self.next_state(state, primitive, 0, False), self.edge_cost(primitive, 0), data

    def _solve_regrasp(self, pose, angle, last_grasp_q):
        for seed in (last_grasp_q, self.problem.start_q, None):
            q = self.problem.arm.solve(pose, angle, seed)
            if q is not None and self.is_clear(pose, angle, q, True):
                return q
        return None

    def _regrasp_edge(self, state, q):
        """Release keeps the last grasp configuration as the seed for the next grasp."""
        pose = self.lattice_to_world(self.state_lattice_pose(state))
        angle = self.door_angle(state.door)
        half_time = self.params.regrasp_time / 2.0
        if state.grasped:
            return state._replace(grasped=False), half_time, EdgeData(q, [pose], [angle], half_time)
        q_grasp = self._solve_regrasp(pose, angle, q)
        if q_grasp is None:
            return None
        return state._replace(grasped=True), half_time, EdgeData(q_grasp, [pose], [angle], half_time)

    def next_state(self, state, primitive, door_steps, grasped):
        if primitive is None:
            return state._replace(door=state.door + door_steps, grasped=grasped)
        heading = (state.heading + primitive.heading_change) % self.params.heading_bins
        return State(state.ix + primitive.cell_offset[0], state.iy + primitive.cell_offset[1], heading,
                     state.door + door_steps, grasped)

    def successors(self, state, data):
        candidates = [self._regrasp_edge(state, data.q)]
        if state.grasped:
            for door_steps in range(1, self.params.max_door_steps + 1):
                candidates.append(self._grasped_edge(state, data.q, None, door_steps))
            for primitive in self.primitives[state.heading]:
                for door_steps in range(0, self.params.max_door_steps + 1):
                    candidates.append(self._grasped_edge(state, data.q, primitive, door_steps))
        else:
            candidates += [self._free_edge(state, data.q, primitive) for primitive in self.primitives[state.heading]]
        return [candidate for candidate in candidates if candidate is not None]

    def plan(self):
        """Return (steps, expansions); steps is a list of PathStep or None."""
        start = State(0, 0, 0, 0, True)
        start_data = EdgeData(self.problem.start_q, [self.problem.start_pose], [self.problem.start_angle], 0.0)
        path, data, expansions = astar(start, start_data, self.is_goal, self.successors, self.heuristic,
                                       self.params.max_expansions)
        if path is None:
            return None, expansions
        steps = [PathStep(data[s].poses, data[s].angles, data[s].duration, s.grasped, data[s].q) for s in path]
        return steps, expansions
