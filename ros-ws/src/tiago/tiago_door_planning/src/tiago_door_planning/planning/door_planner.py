"""A* door-opening planner over (x, y, heading, door angle, grasped, grip).

The lattice is anchored at the robot's start pose, so the start is exactly a lattice state. A grip is an arm on
one face's handle; its feasibility is injected through one ArmChecks per grip, which keeps this module free of
kinematics libraries. A released state keeps the grip that let go; the regrasp may use any grip.
"""

import collections

import numpy as np

from .astar import astar
from .parallel import ForkedMap
from ..model.collision import GRASP, FREE, robot_collides, spheres_in_world
from ..model.door_scene import door_box, wall_boxes
from ..model.geometry import matrix_to_pose2d, pose2d_to_matrix, wrap_angle
from .primitives import primitive_table, turn_drive_turn

PlannerParams = collections.namedtuple(
    "PlannerParams",
    "resolution heading_bins step_cells samples_per_edge angle_step max_door_steps goal_angle linear_speed "
    "angular_speed door_rate regrasp_time max_expansions stall_expansions max_joint_step arm_speed workers",
)

ArmChecks = collections.namedtuple("ArmChecks", "solve spheres max_angle reachable")

Grip = collections.namedtuple("Grip", "arm face")

PlanningProblem = collections.namedtuple(
    "PlanningProblem",
    "scene grid start_pose start_angle start_q free_q grips start_grip start_grasped collision",
)

State = collections.namedtuple("State", "ix iy heading door grasped grip")

EdgeData = collections.namedtuple("EdgeData", "q poses angles duration")

PathStep = collections.namedtuple("PathStep", "poses angles duration grasped q grip")

SPOT_RADIUS = 0.1
DRIVE_HEURISTIC_WEIGHT = 2.0
SPOT_TURN = np.pi / 8.0
ONTO_SPOT_TICK = 0.1
SPOT_SIDESTEP = 0.02
QUICK_ONTO_SPOT_S = 1.5


class Planner(object):
    def __init__(self, problem, params, grasp_spots=None, drive_distance=None, regrasp_in_place=True):
        """With grasp_spots, base poses a released robot can grasp from, the planner only drives (plan_to_spot);
        drive_distance(xy), a lower bound on the distance to drive to a spot, steers it. Without regrasp_in_place a
        held search never releases the handle."""
        self.problem = problem
        self.regrasp_in_place = regrasp_in_place
        self.params = params
        self.grasp_spots = grasp_spots
        self.drive_distance = drive_distance
        self.free_spheres = {}
        self.primitives = primitive_table(params.heading_bins, params.resolution, params.step_cells,
                                          params.samples_per_edge)
        self.grid_start = np.floor(problem.start_angle / params.angle_step + 1e-9) * params.angle_step
        self.goal_door = max(0, int(np.ceil((params.goal_angle - self.grid_start) / params.angle_step - 1e-9)))
        self.walls = wall_boxes(problem.scene)
        self.door_boxes = {}
        self.start_matrix = pose2d_to_matrix(problem.start_pose)
        self.grasped_edges = ForkedMap(self._grasped_edge, params.workers)

    def door_box_at(self, angle):
        key = round(angle, 6)
        if key not in self.door_boxes:
            self.door_boxes[key] = door_box(self.problem.scene, angle)
        return self.door_boxes[key]

    def door_angle(self, door_index):
        """Door angles past the start lie on the angle_step grid, so a search starting between two grid angles
        aims at the same angles as one starting on the grid."""
        first = self.grid_start + self.params.angle_step
        if door_index <= 1:
            return self.problem.start_angle + door_index * (first - self.problem.start_angle)
        return self.grid_start + door_index * self.params.angle_step

    def lattice_to_world(self, lattice_pose):
        return matrix_to_pose2d(self.start_matrix @ pose2d_to_matrix(lattice_pose))

    def state_lattice_pose(self, state):
        heading = 2.0 * np.pi * state.heading / self.params.heading_bins
        return np.array([state.ix * self.params.resolution, state.iy * self.params.resolution, heading])

    def edge_world_poses(self, state, primitive):
        origin = self.state_lattice_pose(state)
        lattice = primitive.samples + np.array([origin[0], origin[1], 0.0])
        return [self.lattice_to_world(pose) for pose in lattice]

    def arm_spheres(self, q, grasped, grip):
        """A released arm is always parked at free_q, so its spheres are computed once."""
        if grasped:
            return self.problem.grips[grip].spheres(q)
        if grip not in self.free_spheres:
            self.free_spheres[grip] = self.problem.grips[grip].spheres(self.problem.free_q)
        return self.free_spheres[grip]

    def is_clear(self, base_pose, door_angle, q, grasped, grip):
        spheres = spheres_in_world(base_pose, self.arm_spheres(q, grasped, grip))
        mode = GRASP if grasped else FREE
        return not robot_collides(self.walls, self.door_box_at(door_angle), self.problem.grid, base_pose, spheres, mode,
                                  self.problem.collision)

    def edge_duration(self, primitive, door_steps):
        params = self.params
        times = [door_steps * params.angle_step / params.door_rate]
        if primitive is not None:
            times += [primitive.length / params.linear_speed, primitive.turn / params.angular_speed]
        return max(times)

    def drive_to_grasp_time(self, state):
        """Lower bound on the time to get within SPOT_RADIUS and SPOT_TURN of a grasp spot; 0 there."""
        if self.grasp_spots is None or len(self.grasp_spots) == 0:
            return 0.0
        pose = self.lattice_to_world(self.state_lattice_pose(state))
        distance = np.linalg.norm(self.grasp_spots[:, :2] - pose[:2], axis=1) - SPOT_RADIUS
        turn = np.abs([wrap_angle(yaw - pose[2]) for yaw in self.grasp_spots[:, 2]]) - SPOT_TURN
        times = np.maximum(distance / self.params.linear_speed, turn / self.params.angular_speed)
        around = 0.0 if self.drive_distance is None else self.drive_distance(pose[:2]) - SPOT_RADIUS
        return max(0.0, float(np.min(times)), around / self.params.linear_speed)

    def heuristic(self, state):
        """A released robot has to drive to a grasp spot and grasp again before the door moves."""
        remaining = max(0, self.goal_door - state.door)
        door_time = remaining * self.params.angle_step / self.params.door_rate
        if state.grasped or remaining == 0:
            return door_time
        return door_time + self.params.regrasp_time / 2.0 + self.drive_to_grasp_time(state)

    def is_goal(self, state):
        return state.door >= self.goal_door

    def _track_grasped(self, poses, angles, q, grip):
        """The tracker moves the arm in small steps, so the plan may not jump to another arm configuration."""
        qs = [q]
        for pose, angle in zip(poses, angles):
            q_next = self.problem.grips[grip].solve(pose, angle, qs[-1])
            if q_next is None or np.max(np.abs(q_next - qs[-1])) > self.params.max_joint_step:
                return None
            if not self.is_clear(pose, angle, q_next, True, grip):
                return None
            qs.append(q_next)
        return qs

    def arm_travel_time(self, qs):
        return float(np.max(np.sum(np.abs(np.diff(qs, axis=0)), axis=0))) / self.params.arm_speed

    def _primitive(self, state, index):
        return None if index is None else self.primitives[state.heading][index]

    def _grasped_moves(self, state, q, may_improve):
        """(state, q, primitive index, door steps) of the grasped edges whose shortest duration could still improve
        on the state they reach; an index of None turns the door without moving the base."""
        moves = [(None, steps) for steps in range(1, self.params.max_door_steps + 1)]
        moves += [(index, steps) for index in range(len(self.primitives[state.heading]))
                  for steps in range(0, self.params.max_door_steps + 1)]
        return [(state, q, index, steps) for index, steps in moves
                if may_improve(self.next_state(state, self._primitive(state, index), steps, True),
                               self.edge_duration(self._primitive(state, index), steps))]

    def _grasped_edge(self, move):
        """None when the edge is infeasible."""
        state, q, index, door_steps = move
        primitive = self._primitive(state, index)
        next_state = self.next_state(state, primitive, door_steps, True)
        start_pose = self.lattice_to_world(self.state_lattice_pose(state))
        poses = self.edge_world_poses(state, primitive) if primitive else [start_pose] * self.params.samples_per_edge
        fractions = np.linspace(0.0, 1.0, len(poses) + 1)[1:]
        angles = [self.door_angle(state.door + door_steps * f) for f in fractions]
        qs = self._track_grasped(poses, angles, q, state.grip)
        if qs is None:
            return None
        duration = max(self.edge_duration(primitive, door_steps), self.arm_travel_time(qs))
        return next_state, duration, EdgeData(qs[-1], poses, angles, duration)

    def _free_edge(self, state, last_grasp_q, primitive, may_improve):
        next_state = self.next_state(state, primitive, 0, False)
        duration = self.edge_duration(primitive, 0)
        if not may_improve(next_state, duration):
            return None
        poses = self.edge_world_poses(state, primitive)
        angle = self.door_angle(state.door)
        if not all(self.is_clear(pose, angle, self.problem.free_q, False, state.grip) for pose in poses):
            return None
        return next_state, duration, EdgeData(last_grasp_q, poses, [angle] * len(poses), duration)

    def _solve_regrasp(self, pose, state, seeds, grip):
        """A grasp is only worth its time when the grip can also move the door one more step."""
        solve = self.problem.grips[grip].solve
        angle = self.door_angle(state.door)
        next_angle = angle if self.is_goal(state) else self.door_angle(state.door + 1)
        for seed in seeds:
            q_next = solve(pose, next_angle, seed)
            q = None if q_next is None else solve(pose, angle, q_next)
            if q is not None and self.is_clear(pose, angle, q, True, grip) and self.problem.grips[grip].reachable(
                pose, angle, q
            ):
                return q
        return None

    def _regrasp_seeds(self, state, grip, last_grasp_q):
        if grip != state.grip:
            return self.problem.start_q, None
        return last_grasp_q, self.problem.start_q, None

    def _regrasp_edges(self, state, q):
        """Release keeps the last grasp configuration as the seed for the next grasp with the same grip."""
        pose = self.lattice_to_world(self.state_lattice_pose(state))
        angle = self.door_angle(state.door)
        half_time = self.params.regrasp_time / 2.0
        if state.grasped:
            if not self.regrasp_in_place:
                return []
            return [(state._replace(grasped=False), half_time, EdgeData(q, [pose], [angle], half_time))]
        if self.grasp_spots is not None:
            return []
        edges = []
        for grip in self.problem.grips:
            q_grasp = self._solve_regrasp(pose, state, self._regrasp_seeds(state, grip, q), grip)
            if q_grasp is not None:
                grasped = state._replace(grasped=True, grip=grip)
                edges.append((grasped, half_time, EdgeData(q_grasp, [pose], [angle], half_time)))
        return edges

    def next_state(self, state, primitive, door_steps, grasped):
        if primitive is None:
            return state._replace(door=state.door + door_steps, grasped=grasped)
        heading = (state.heading + primitive.heading_change) % self.params.heading_bins
        ix, iy = state.ix + primitive.cell_offset[0], state.iy + primitive.cell_offset[1]
        return State(ix, iy, heading, state.door + door_steps, grasped, state.grip)

    def successors(self, state, data, may_improve):
        candidates = self._regrasp_edges(state, data.q)
        if state.grasped:
            candidates += self.grasped_edges(self._grasped_moves(state, data.q, may_improve))
        else:
            candidates += [self._free_edge(state, data.q, primitive, may_improve) for primitive in self.primitives[state.heading]]
        return [candidate for candidate in candidates if candidate is not None]

    def _search(self, is_goal, progress, stall_limit, heuristic_weight=1.0):
        start = State(0, 0, 0, 0, self.problem.start_grasped, self.problem.start_grip)
        start_data = EdgeData(self.problem.start_q, [self.problem.start_pose], [self.problem.start_angle], 0.0)
        try:
            path, data, _ = astar(start, start_data, is_goal, self.successors,
                                  lambda state: heuristic_weight * self.heuristic(state), self.params.max_expansions,
                                  progress, stall_limit)
        finally:
            self.grasped_edges.close()
        if path is None:
            return None
        return [PathStep(data[s].poses, data[s].angles, data[s].duration, s.grasped, data[s].q, s.grip) for s in path]

    def plan(self):
        """PathSteps to the goal, or to the widest door angle reached when the search runs out; it gives up after
        stall_expansions expansions without opening the door further."""
        return self._search(self.is_goal, lambda state: state.door, self.params.stall_expansions)

    def _spots_in_reach(self, pose):
        distance = np.linalg.norm(self.grasp_spots[:, :2] - pose[:2], axis=1)
        turn = np.abs([wrap_angle(yaw - pose[2]) for yaw in self.grasp_spots[:, 2]])
        near = (distance <= SPOT_RADIUS + 1e-9) & (turn <= SPOT_TURN + 1e-9)
        return self.grasp_spots[near][np.argsort(distance[near])]

    def _onto_spot(self, pose, angle, grip, max_duration=np.inf):
        """The quickest clear step, of at most max_duration, from a pose onto a grasp spot next to it, or None."""
        best = None
        for spot in self._spots_in_reach(pose):
            for poses, duration in self._steps_onto(pose, spot):
                quicker = duration <= max_duration and (best is None or duration < best.duration)
                if quicker and all(self.is_clear(p, angle, self.problem.free_q, False, grip) for p in poses):
                    best = PathStep(poses, [angle] * len(poses), duration, False, None, grip)
        return best

    def _steps_onto(self, pose, spot):
        """(poses, duration) of the ways onto a spot: turn, drive straight and turn, sampled one tick apart so the
        timing survives resampling; or one straight glide whose sideways part, which the base cannot drive, stays
        within what the spot tolerates."""
        params = self.params
        poses, distance, turn = turn_drive_turn(pose, spot, params.linear_speed * ONTO_SPOT_TICK,
                                                params.angular_speed * ONTO_SPOT_TICK)
        steps = [(poses, distance / params.linear_speed + turn / params.angular_speed)]
        change = np.array([*(spot[:2] - pose[:2]), wrap_angle(spot[2] - pose[2])])
        if abs(change[:2] @ np.array([-np.sin(pose[2]), np.cos(pose[2])])) <= SPOT_SIDESTEP:
            fractions = np.linspace(0.0, 1.0, params.samples_per_edge + 1)[1:]
            glide = max(np.linalg.norm(change[:2]) / params.linear_speed, abs(change[2]) / params.angular_speed)
            steps.append(([pose + fraction * change for fraction in fractions], glide))
        return steps

    def _state_onto_spot(self, state, max_duration):
        pose = self.lattice_to_world(self.state_lattice_pose(state))
        return self._onto_spot(pose, self.door_angle(state.door), state.grip, max_duration)

    def plan_to_spot(self):
        """PathSteps that drive a released robot exactly onto one of the grasp spots, or None. The drive only needs
        to be short, not shortest, so its heuristic is inflated by DRIVE_HEURISTIC_WEIGHT. The search's cost leaves
        out the last step onto the spot, so a quick one is sought first."""
        for max_duration in (QUICK_ONTO_SPOT_S, np.inf):
            steps = self._drive_to_spot(max_duration)
            if steps is not None:
                return steps
        return None

    def _drive_to_spot(self, max_duration):
        steps = self._search(lambda state: self._state_onto_spot(state, max_duration) is not None, None, None,
                             DRIVE_HEURISTIC_WEIGHT)
        if steps is None:
            return None
        last = steps[-1]
        onto = self._onto_spot(last.poses[-1], last.angles[-1], last.grip, max_duration)
        return steps + [onto._replace(q=last.q)]
