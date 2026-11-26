#!/usr/bin/env python3
"""Action server opening a door: plan the grasp and the opening, grasp, then track the grasped segments"""

import collections
import threading

import actionlib
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from tiago_door_planning.motion.arm_moves import (
    Surroundings,
    arm_path_clear,
    base_path_leaves_clear,
    grasp_motion,
)
from tiago_door_planning.motion.arm_moves import path_leaves_clear, release_paths
from tiago_door_planning.model.collision import FREE, GRASP, clear_under_robot, distance_grid
from tiago_door_planning.planning.door_planner import Grip, PlanningProblem
from tiago_door_planning.planning.approach import approach_spots
from tiago_door_planning.planning.door_goal import passable_angle
from tiago_door_planning.planning.pass_plan import plan_pass
from tiago_door_planning.sensing.door_sensing import door_estimate, hand_off_door
from tiago_door_planning.model.geometry import (
    make_pose,
    rotation_to_quaternion,
    wrap_angle,
)
from tiago_door_planning.motion.grasp_motion import (
    joint_path,
    line_path,
    retreat_displacement,
    toward_robot,
)
from tiago_door_planning.motion.grasp_motion import (
    lateral_error,
    unrolled_offset,
    via_path,
)
from tiago_door_planning.planning.grips import (
    arm_checks,
    describe_segment,
    door_scenes,
    face_normals,
    grasp_offsets,
)
from tiago_door_planning.planning.grips import (
    grasp_action,
    other_side,
    plan_cost,
    segment_grip,
)
from tiago_door_planning.model.kinematics import (
    gripper_pose,
    link_spheres,
    velocity_limits,
)
from tiago_door_planning.ros.config import (
    apply_pipeline_speed,
    collision_params,
    goal_params,
    planner_params,
)
from tiago_door_planning.msg import (
    ExecuteDoorSegmentAction,
    ExecuteDoorSegmentGoal,
    OpenDoorAction,
    PassDoorAction,
)
from tiago_door_planning.msg import (
    OpenDoorFeedback,
    OpenDoorResult,
    PassDoorFeedback,
    PassDoorResult,
)
from tiago_door_planning.srv import (
    MeasureDoor,
    MeasureDoorResponse,
    PlanApproach,
    PlanApproachResponse,
    TrackDoorLeaf,
    TrackDoorLeafRequest,
)
from tiago_door_planning.planning.opening_plan import plan_opening, stop_angle
from tiago_door_planning.planning.plan_output import plan_to_segments
from tiago_door_planning.ros.robot import (
    SCAN_TOPIC,
    SIDES,
    LocalizationLost,
    OpeningFailed,
    Robot,
    SegmentFailed,
)
from tiago_door_planning.ros.ros_conversions import pose_msg_to_matrix
from tiago_door_planning.sensing.localization import is_localization_lost
from tiago_door_planning.motion.tracking import (
    door_angle_from_gripper,
    measure_grasp_offset,
)
from tiago_door_planning.motion.trajectory import (
    from_angle,
    split_by_sweep,
    straight_drive,
)

SEGMENT_TIMEOUT_MARGIN_S = 10.0
PASSABLE_ANGLE_STEP = np.radians(1.0)

OpeningPlan = collections.namedtuple("OpeningPlan", "move approach segments")
DoorProgress = collections.namedtuple("DoorProgress", "angle regrasps")


class DoorOpen(Exception):
    """The door is open wide enough to pass."""


def describe_reading(reading):
    if reading is None:
        return "none"
    fitted = (
        "unseen" if reading.angle is None else "%.1f deg" % np.degrees(reading.angle)
    )
    return "%s, open at least %.1f deg" % (fitted, np.degrees(reading.at_least))


def rounded(values):
    return None if values is None else np.round(values, 3).tolist()


def pose_msg(frame, pose):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3, 3]
    orientation = msg.pose.orientation
    orientation.x, orientation.y, orientation.z, orientation.w = rotation_to_quaternion(
        pose[:3, :3]
    )
    return msg


def occupancy_to_grid(msg, occupied_threshold, base_xy, clear_radius):
    """The distance grid, with the cells under the robot freed; (grid, number of occupied cells freed)."""
    info = msg.info
    occupancy = np.array(msg.data, dtype=float).reshape(info.height, info.width)
    origin = [info.origin.position.x, info.origin.position.y]
    occupancy, cleared = clear_under_robot(
        occupancy, info.resolution, origin, base_xy, clear_radius
    )
    return distance_grid(occupancy, info.resolution, origin, occupied_threshold), cleared


def segment_goal(goal, scene, trajectory, grasped, action, free_wrench):
    segment = ExecuteDoorSegmentGoal()
    segment.hinge_pose = goal.hinge_pose
    segment.handle_pose = pose_msg(
        goal.handle_pose.header.frame_id, scene.handle_closed
    )
    segment.hinge_side, segment.push, segment.arm = (
        goal.hinge_side,
        goal.push,
        trajectory.side,
    )
    segment.grasped, segment.grasp_action = grasped, action if grasped else ""
    segment.free_wrench = list(free_wrench) if grasped and free_wrench is not None else []
    segment.base_x, segment.base_y, segment.base_yaw = trajectory.base_poses.T.tolist()
    segment.door_angles = trajectory.door_angles.tolist()
    segment.sample_period = trajectory.sample_period
    segment.arm_qs = trajectory.arm_qs.ravel().tolist()
    return segment


class DoorOpeningServer(object):
    """open_door and pass_door, plus the plan_approach and measure_door services; one request at a time, since they
    share the robot and the door set up for the request."""

    def __init__(self):
        self._config = config = apply_pipeline_speed(rospy.get_param("~"))
        self._grasp_config = config["grasp"]
        self._collision = collision_params(config)
        self._move_collision = self._collision._replace(
            margin=config["grasp"]["move_margin"]
        )
        self._max_sweep = np.radians(config["grasp"]["max_sweep_deg"])
        self._robot = Robot(config, with_moves=True)
        self._busy = threading.Lock()
        self._active = None
        self._segment_client = actionlib.SimpleActionClient(
            "execute_door_segment", ExecuteDoorSegmentAction
        )
        self._server = actionlib.SimpleActionServer(
            "open_door", OpenDoorAction, self._execute, auto_start=False
        )
        self._pass_server = actionlib.SimpleActionServer(
            "pass_door", PassDoorAction, self._execute_pass, auto_start=False
        )
        rospy.Service("plan_approach", PlanApproach, self._on_plan_approach)
        rospy.Service("measure_door", MeasureDoor, self._on_measure_door)
        self._track_leaf = rospy.ServiceProxy("track_door_leaf", TrackDoorLeaf)
        self._server.start()
        self._pass_server.start()

    # The actions and services

    def _execute(self, goal):
        result = OpenDoorResult()
        result.success, result.message = self._run_exclusive(
            self._server, goal, self._open, "door opened"
        )
        result.final_door_angle = self._door_angle
        rospy.loginfo(
            "[DoorOpening] %s (door angle %.1f deg)",
            result.message,
            np.degrees(self._door_angle),
        )
        self._finish(self._server, result)

    def _execute_pass(self, goal):
        result = PassDoorResult()
        result.success, result.message = self._run_exclusive(
            self._pass_server, goal, self._pass, "passed the door"
        )
        rospy.loginfo("[DoorOpening] %s", result.message)
        self._finish(self._pass_server, result)

    def _run_exclusive(self, server, goal, run, done):
        """(success, message) of run(); any error, not only an OpeningFailed, ends the goal with its message."""
        if not self._busy.acquire(False):
            return False, "busy with another door request"
        self._active = server
        self._start_request(goal)
        self._mask_leaf(goal, True)
        try:
            return (True, run() or done)
        except OpeningFailed as error:
            return False, str(error)
        except Exception as error:  # a goal must always end with a result
            rospy.logerr("[DoorOpening] %s", error, exc_info=True)
            return False, "internal error: %s" % error
        finally:
            self._mask_leaf(goal, False)
            self._active = None
            self._busy.release()

    def _mask_leaf(self, goal, enable):
        """Asks door_leaf_tracker to hide the moving leaf from the SLAM scans; the door request goes on without it."""
        if not self._config["leaf_mask"]["enabled"]:
            return
        request = TrackDoorLeafRequest(
            user="door_opening",
            enable=enable,
            hinge_pose=goal.hinge_pose,
            handle_pose=goal.handle_pose,
            hinge_side=goal.hinge_side,
            push=goal.push,
        )
        try:
            self._track_leaf.wait_for_service(1.0)
            self._track_leaf(request)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logwarn("[DoorOpening] leaf mask %s failed: %s", "on" if enable else "off", error)

    @staticmethod
    def _finish(server, result):
        if result.message == "preempted":
            server.set_preempted(result)
        elif result.success:
            server.set_succeeded(result)
        else:
            server.set_aborted(result)

    def _start_request(self, goal):
        self._goal, self._push = goal, goal.push
        self._door_angle, self._passable_angle, self._laser_decided = 0.0, np.inf, False
        self._goals = None
        self._holding, self._holding_action, self._approach = None, None, None
        self._free_wrench = None
        self._released_q = None
        self._last_grip = None

    def _stage(self, text):
        rospy.loginfo("[DoorOpening] %s", text)
        if self._active is None:
            return
        feedback = (
            OpenDoorFeedback() if self._active is self._server else PassDoorFeedback()
        )
        feedback.stage = text
        self._active.publish_feedback(feedback)
        self._raise_if_preempted()

    def _raise_if_preempted(self):
        if self._active is not None and self._active.is_preempt_requested():
            raise OpeningFailed("preempted")

    def _open(self):
        """Ends holding the handle, or released with the arm parked when the goal asks for release_at_end; a plan-only
        goal returns the plan's description."""
        if not self._goal.plan_only:
            self._robot.require_wrench(self._grasp_config["arms"])
        self._stage("planning")
        self._set_up_door()
        plan = self._choose_plan()
        if self._goal.plan_only:
            return "planned: %s" % "; ".join(describe_segment(s) for s in plan.segments)
        grip = self._last_grip = segment_grip(plan.segments[0])
        if plan.move is None or self._starts_released(plan):
            self._released_q = None if plan.approach is None else plan.approach[-1]
        else:
            self._stage("grasp 1 with the %s arm" % grip.arm)
            self._grasp(
                grip.arm, plan.move, plan.approach, grasp_action(self._push, grip.face)
            )
            self._measure_door(self._scenes["front"], self._offsets[grip])
        try:
            self._open_with_recoveries(plan.segments)
        except LocalizationLost as error:
            rospy.logerr("[DoorOpening] %s; holding still, the operator takes over", error)
            raise
        except OpeningFailed as error:
            if self._goal.release_at_end and str(error) != "preempted":
                self._let_go_after_failure()
            raise
        if self._goal.release_at_end:
            self._let_go(self._scenes[self._last_grip.face], self._last_grip.arm)
        return None

    def _let_go_after_failure(self):
        """Best effort: a failed opening leaves the hand open and the arm parked when it can."""
        if self._last_grip is None:
            return
        try:
            self._let_go(self._scenes[self._last_grip.face], self._last_grip.arm)
        except Exception as error:  # the failure being reported matters more
            rospy.logwarn("[DoorOpening] could not let go after the failure: %s", error)

    def _starts_released(self, plan):
        """A plan that lets go of the handle before moving the door needs no first grasp; the robot drives first."""
        return (
            self._grasp_config["skip_unused_first_grasp"]
            and not plan.segments[0].grasped
        )

    def _set_up_door(self):
        goal = self._goal
        self._frame = goal.hinge_pose.header.frame_id
        self._scenes = door_scenes(
            pose_msg_to_matrix(goal.hinge_pose.pose),
            pose_msg_to_matrix(goal.handle_pose.pose),
            goal.hinge_side,
            goal.push,
            self._config,
        )
        self._arms = {side: self._robot.load_arm(side) for side in SIDES}
        self._home = {side: self._robot.q(side) for side in SIDES}
        self._grid = self._costmap()
        self._normals = face_normals(self._scenes, self._base_pose())
        self._offsets = grasp_offsets(
            self._config, self._push, self._scenes, self._normals
        )

    def _on_plan_approach(self, request):
        response = PlanApproachResponse()
        if not self._busy.acquire(False):
            response.message = "busy with another door request"
            return response
        try:
            self._start_request(request)
            self._set_up_door()
            approaches = self._approach_spots()
            response.poses = [
                pose_msg(self._frame, make_pose([a.spot.pose[0], a.spot.pose[1], 0.0], a.spot.pose[2]))
                for a in approaches
            ]
            response.success = bool(approaches)
            response.message = (
                "%d base poses open the door, the best with %d grasps in %.0f s"
                % (len(approaches), approaches[0].grasps, approaches[0].duration)
                if approaches
                else "no base pose near the handle opens the door"
            )
        except Exception as error:  # the caller gets the reason
            rospy.logerr("[DoorOpening] %s", error, exc_info=True)
            response.message = "internal error: %s" % error
        finally:
            self._busy.release()
        rospy.loginfo("[DoorOpening] plan_approach: %s", response.message)
        return response

    def _approach_spots(self):
        """Base poses to grasp the closed door from, each with an opening planned from it, best first."""
        side = self._grasp_config["arms"][0]
        problem = self._problem(
            self._base_pose(), self._offsets, Grip(side, "front"), self._robot.q(side), False
        )
        started = rospy.get_time()
        approaches = approach_spots(
            problem,
            planner_params(self._config, self._push),
            goal_params(self._config),
            self._config["goal"]["approach_plans"],
            lambda text: rospy.loginfo("[DoorOpening] approach: %s", text),
        )
        rospy.loginfo(
            "[DoorOpening] approach planning took %.1f s: %s",
            rospy.get_time() - started,
            "; ".join(
                "%s (%d grasps, %.0f s)" % (rounded(a.spot.pose), a.grasps, a.duration)
                for a in approaches
            ),
        )
        return approaches

    def _on_measure_door(self, request):
        response = MeasureDoorResponse()
        if not self._busy.acquire(False):
            response.message = "busy with another door request"
            return response
        try:
            self._start_request(request)
            self._frame = request.hinge_pose.header.frame_id
            self._scenes = door_scenes(
                pose_msg_to_matrix(request.hinge_pose.pose),
                pose_msg_to_matrix(request.handle_pose.pose),
                request.hinge_side,
                request.push,
                self._config,
            )
            reading = self._door_reading(self._scenes["front"])
            if reading is None:
                response.message = "no scan"
            else:
                response.seen = reading.angle is not None
                response.angle = reading.angle if response.seen else 0.0
                response.at_least = reading.at_least
                response.message = describe_reading(reading)
        except Exception as error:  # the caller gets the reason
            rospy.logerr("[DoorOpening] %s", error, exc_info=True)
            response.message = "internal error: %s" % error
        finally:
            self._busy.release()
        return response

    def _pass(self):
        """Drives the released robot through the doorway past the leaf where the opening left it."""
        self._stage("planning the way through the door")
        self._set_up_door()
        self._door_angle = self._goal.door_angle
        scene = self._scenes["front"]
        self._door_seen_moved(scene)
        steps = self._pass_steps(self._base_pose(), self._door_angle)
        if steps is None:
            steps = self._pass_after_stepping_away(scene)
        if steps is None:
            raise OpeningFailed(
                "no way through the door with the leaf at %.0f deg" % np.degrees(self._door_angle)
            )
        planner = self._config["planner"]
        for segment in plan_to_segments(steps, planner["drive_ramp_s"]):
            self._stage("driving through the door")
            self._run_segment(scene, segment.trajectory, False)
        return None

    def _pass_steps(self, base_pose, door_angle):
        """The released robot's way from base_pose through the doorway past the leaf at door_angle, or None."""
        side = self._grasp_config["arms"][0]
        problem = self._problem(
            base_pose, self._offsets, Grip(side, "front"), self._home[side], False
        )
        return plan_pass(
            problem._replace(start_angle=door_angle),
            planner_params(self._config, self._push)._replace(step_cells=self._config["pass"]["step_cells"]),
            goal_params(self._config),
            self._config["pass"]["distances"],
            self._push,
        )

    def _pass_after_stepping_away(self, scene):
        """A base left within the collision margin of the leaf has no way out from where it stands: a straight step
        away from the leaf, or back the other way, from whose end a way through is planned, comes first."""
        side = self._grasp_config["arms"][0]
        base = self._base_pose()
        heading = np.array([np.cos(base[2]), np.sin(base[2])])
        away = np.copysign(
            self._grasp_config["standoff"],
            heading @ toward_robot(scene, self._door_angle, base[:2]),
        )
        for distance in (away, -away):
            drive = self._clear_straight_drive(scene, side, base, distance)
            if drive is None or self._pass_steps(drive.base_poses[-1], self._door_angle) is None:
                continue
            self._stage("stepping the base away from the leaf")
            self._run_segment(scene, drive, False)
            return self._pass_steps(self._base_pose(), self._door_angle)
        return None

    def _open_with_recoveries(self, segments):
        """A recovery that fails itself, say when the arm cannot fold, counts as the next failed segment."""
        holding_from_plan = True
        recoveries = self._config["goal"]["recoveries"]
        while True:
            try:
                if segments is None:
                    segments, holding_from_plan = self._plan_from_here(), False
                self._follow(segments, holding_from_plan)
                self._require_passable()
                return
            except DoorOpen:
                rospy.loginfo(
                    "[DoorOpening] door at %.0f deg, open wide enough to pass",
                    np.degrees(self._door_angle),
                )
                return
            except SegmentFailed as error:
                self._door_seen_moved(self._scenes["front"])
                if self._wide_enough():
                    rospy.logwarn(
                        "[DoorOpening] %s; the door is open wide enough to pass", error
                    )
                    return
                if recoveries == 0:
                    raise
                recoveries -= 1
                rospy.logwarn("[DoorOpening] %s; planning again from here", error)
                segments = None

    def _require_passable(self):
        if not self._wide_enough():
            raise SegmentFailed(
                "the plan ended with the door at %.0f deg, short of the %.0f deg to pass"
                % (np.degrees(self._door_angle), np.degrees(self._passable_angle))
            )

    def _stop_if_passable(self):
        """Every further grasp costs a release, a drive and a regrasp, which a door open wide enough does not need."""
        if self._passable():
            raise DoorOpen()

    def _passable(self):
        return self._sure_door_angle() >= self._passable_angle

    def _sure_door_angle(self):
        """A door angle the laser fit gave against the hand counts less the fit's error."""
        margin = (
            np.radians(self._config["scan"]["fit_error_deg"])
            if self._laser_decided
            else 0.0
        )
        return self._door_angle - margin

    def _wide_enough(self):
        """At the passable angle, or short of it with a way through the doorway planned: the goal is the doorway's
        passable angle rounded up to the planner's step plus a margin, which a door a few degrees short may not need.
        Below the doorway's own passable angle no way through is looked for."""
        if self._passable():
            return True
        angle = self._sure_door_angle()
        goal = goal_params(self._config)
        needed = passable_angle(self._scenes["front"], goal.pass_width, goal.handle_depth, PASSABLE_ANGLE_STEP)
        if needed is None or angle < needed:
            return False
        self._grid = self._costmap()
        if self._pass_steps(self._base_pose(), angle) is None:
            rospy.loginfo("[DoorOpening] no way through the door planned at %.0f deg", np.degrees(angle))
            return False
        rospy.loginfo(
            "[DoorOpening] door at %.0f deg, short of %.0f deg, but a way through is planned",
            np.degrees(angle),
            np.degrees(self._passable_angle),
        )
        return True

    # Planning

    def _choose_plan(self):
        plans, failures = [], []
        for side in self._grasp_config["arms"]:
            try:
                plans.append(self._plan_for(side))
            except OpeningFailed as error:
                failures.append("%s arm: %s" % (side, error))
                rospy.loginfo("[DoorOpening] %s", failures[-1])
        if not plans:
            raise OpeningFailed("; ".join(failures))
        best = min(plans, key=lambda plan: plan_cost(plan.segments, self._max_sweep))
        rospy.loginfo(
            "[DoorOpening] %d grasps: %s",
            plan_cost(best.segments, self._max_sweep)[0],
            "; ".join(describe_segment(s) for s in best.segments),
        )
        return best

    def _plan_for(self, side):
        """From a base the arm cannot grasp from, the plan starts released and drives onto a grasp spot first."""
        self._stage("planning with the %s arm" % side)
        grip = Grip(side, "front")
        base_pose = self._base_pose()
        try:
            move, approach = self._plan_grasp(self._scenes["front"], grip, None)
        except OpeningFailed as error:
            rospy.loginfo("[DoorOpening] %s arm: %s here; driving to a grasp first", side, error)
            return OpeningPlan(
                None,
                None,
                self._plan_opening(
                    base_pose,
                    self._offsets,
                    grip,
                    self._robot.q(side),
                    grasped=False,
                    first_spots=self._first_spots(base_pose, grip),
                ),
            )
        return OpeningPlan(
            move,
            approach,
            self._plan_opening(base_pose, self._offsets, grip, approach[-1]),
        )

    def _first_spots(self, base_pose, grip):
        """The spots plan_approach would send the robot to; the nearest grasp spots may not open the door far."""
        approaches = approach_spots(
            self._problem(base_pose, self._offsets, grip, self._robot.q(grip.arm), False),
            planner_params(self._config, self._push),
            goal_params(self._config),
            self._config["goal"]["approach_plans"],
        )
        return [approach.spot for approach in approaches] or None

    def _problem(self, base_pose, offsets, grip, start_q, grasped):
        checks = arm_checks(
            self._config,
            self._push,
            self._scenes,
            offsets,
            self._normals,
            self._arms,
            self._home,
        )
        return PlanningProblem(
            self._scenes["front"],
            self._grid,
            base_pose,
            self._door_angle,
            start_q,
            self._home[grip.arm],
            checks,
            grip,
            grasped,
            self._collision,
        )

    def _plan_opening(self, base_pose, offsets, grip, grasp_q, grasped=True, first_spots=None):
        """Replans keep the goal angles of the first plan; every plan reads the costmap afresh."""
        self._grid = self._costmap()
        problem = self._problem(base_pose, offsets, grip, grasp_q, grasped)
        started = rospy.get_time()
        steps, self._goals = plan_opening(
            problem,
            planner_params(self._config, self._push),
            goal_params(self._config),
            self._goals,
            first_spots,
        )
        rospy.loginfo(
            "[DoorOpening] planning took %.1f s%s",
            rospy.get_time() - started,
            "" if steps is None else ", the plan ends at %.1f deg" % np.degrees(steps[-1].angles[-1]),
        )
        if self._goals is None:
            raise OpeningFailed("door cannot be opened wide enough to pass from here")
        if steps is None:
            raise OpeningFailed(
                "no opening plan to %.0f deg" % np.degrees(self._goals[0])
            )
        self._set_stop_angle(steps)
        planner = self._config["planner"]
        held_ramp = planner["push_release_ramp_s"] if self._push else 0.0
        return plan_to_segments(steps, planner["drive_ramp_s"], held_ramp)

    def _plan_grasp(self, scene, grip, planned_q):
        """The arm unfolds from where it is parked folded through grasp/unfold_q, since unfolding straight to the
        handle swings it into the leaf."""
        unfold_q = np.array(self._grasp_config["unfold_q"])
        motion = grasp_motion(
            self._arms[grip.arm],
            self._surroundings(scene, grip.arm),
            self._offsets[grip],
            self._robot.q(grip.arm),
            planned_q,
            unfold_q,
            self._grasp_config,
            self._move_collision,
        )
        if motion is None:
            raise OpeningFailed("no collision-free arm motion to the handle")
        return motion

    def _plan_from_here(self):
        """Lets go of the door and plans the rest of the opening from where the robot and the door really are."""
        if self._holding is not None:
            self._release(self._scenes[self._last_grip.face])
        self._park(self._scenes["front"], self._last_grip.arm)
        self._stage("planning again from %.0f deg" % np.degrees(self._door_angle))
        return self._plan_released()

    def _set_stop_angle(self, steps):
        """Every plan sets where the opening stops, the plan's own end when it settled short of the goal."""
        target = stop_angle(self._goals, steps)
        if target < self._goals[0] - 1e-9:
            rospy.loginfo(
                "[DoorOpening] the last turn to %.0f deg is not reachable, ending at %.0f deg",
                np.degrees(self._goals[0]),
                np.degrees(target),
            )
        if target != self._passable_angle:
            rospy.loginfo(
                "[DoorOpening] opening to %.0f deg, grasp poses up to %.0f deg",
                np.degrees(target),
                np.degrees(self._goals[1]),
            )
        self._passable_angle = target

    def _plan_released(self):
        """The arm configuration that last held the handle seeds the next grasps, as it does in the first plan."""
        q = (
            self._robot.q(self._last_grip.arm)
            if self._released_q is None
            else self._released_q
        )
        return self._plan_opening(
            self._base_pose(), self._offsets, self._last_grip, q, grasped=False
        )

    # Following the plan

    def _follow(self, segments, holding_from_plan):
        segments = list(segments)
        progress = DoorProgress(self._door_angle, 0)
        while segments:
            segment = segments.pop(0)
            if segment.grasped:
                self._open_grasped(segment, holding_from_plan)
                holding_from_plan = False
            else:
                self._stop_if_passable()
                progress = self._require_progress(progress)
                self._let_go(self._scenes["front"], segment.trajectory.side)
                if self._door_seen_moved(self._scenes["front"]):
                    self._stop_if_passable()
                    segments = self._replan_released([segment] + segments)
                    continue
                if not segments:
                    raise SegmentFailed(
                        "the plan drives on without a grasp, the door at %.0f deg"
                        % np.degrees(self._door_angle)
                    )
                self._drive_released(self._scenes["front"], segment.trajectory)
                segments = self._regrasp_and_replan(segments)
                holding_from_plan = True

    def _require_progress(self, progress):
        """A regrasp that leaves the door where the last one did can repeat forever; a few in a row fail the segment so
        the recovery plans again."""
        goal = self._config["goal"]
        if self._door_angle >= progress.angle + np.radians(goal["min_progress_deg"]):
            return DoorProgress(self._door_angle, 0)
        if progress.regrasps >= goal["regrasps_without_progress"]:
            raise SegmentFailed(
                "%d regrasps left the door at %.0f deg"
                % (progress.regrasps, np.degrees(self._door_angle))
            )
        return progress._replace(regrasps=progress.regrasps + 1)

    def _open_grasped(self, segment, first):
        """Every piece but the very first starts from a fresh grasp. A door a little past a piece's first angle still
        gets the base motion planned there; the tracker catches up with the door."""
        grip = self._last_grip = segment_grip(segment)
        scene = self._scenes[grip.face]
        for index, piece in enumerate(
            split_by_sweep(segment.trajectory, self._max_sweep)
        ):
            if self._door_angle >= piece.door_angles[-1]:
                continue
            piece = self._from_door_angle(piece)
            if self._holding is not None and (index > 0 or not first):
                self._stop_if_passable()
                self._release(scene)
            if self._holding is None:
                self._stop_if_passable()
                self._grasp_handle(scene, grip, piece.arm_qs[0])
                self._stop_if_passable()
            self._stage(
                "opening to %.0f deg with the %s arm on the %s handle"
                % (np.degrees(piece.door_angles[-1]), grip.arm, grip.face)
            )
            self._run_segment(scene, piece, True)

    def _let_go(self, scene, side):
        if self._holding is not None:
            self._release(scene)
        self._park(scene, side)

    def _door_seen_moved(self, scene):
        """The laser seeing the door away from where the hand last had it moves the door there: a push grasp cannot
        brake the door, so it swings on after the hand lets go, and a segment may fail with the door already open."""
        tolerance = np.radians(self._grasp_config["door_agreement_deg"])
        reading = self._door_reading(scene)
        angle = door_estimate(self._door_angle, reading, tolerance)
        if angle == self._door_angle:
            return False
        rospy.loginfo(
            "[DoorOpening] the laser sees the door at %.1f deg (laser %s)",
            np.degrees(angle),
            describe_reading(reading),
        )
        self._door_angle, self._laser_decided = angle, True
        return True

    def _replan_released(self, remaining):
        """The rest of the opening for the door where it is now; when that finds nothing the robot follows the plan
        it has, and the next grasp measures the door again."""
        self._stage(
            "planning again for the door at %.0f deg" % np.degrees(self._door_angle)
        )
        try:
            return self._plan_released()
        except OpeningFailed as error:
            rospy.logwarn("[DoorOpening] %s; following the previous plan", error)
            return remaining

    def _drive_released(self, scene, trajectory):
        self._stage("driving to the next grasp")
        self._run_segment(scene, trajectory, False)

    def _regrasp_and_replan(self, remaining):
        """A released drive ends a few centimetres off its plan, so the rest is planned again from the new grasp, with
        the hand where it really holds the handle; when that finds nothing the robot follows the plan it has.
        """
        grip = segment_grip(remaining[0])
        held = self._grasp_handle(
            self._scenes[grip.face], grip, remaining[0].trajectory.arm_qs[0]
        )
        self._stop_if_passable()
        if self._near_plan(remaining[0].trajectory):
            return remaining
        measured = dict(self._offsets)
        measured[grip] = unrolled_offset(self._offsets[grip], held)
        self._stage("planning from the new grasp")
        try:
            return self._plan_opening(
                self._base_pose(), measured, grip, self._robot.q(grip.arm)
            )
        except OpeningFailed as error:
            rospy.logwarn("[DoorOpening] %s; following the previous plan", error)
            return remaining

    def _near_plan(self, trajectory):
        """A base standing where the plan holds the door at its current angle needs no new plan."""
        goal = self._config["goal"]
        error = self._base_pose() - self._from_door_angle(trajectory).base_poses[0]
        distance, turn = np.linalg.norm(error[:2]), wrap_angle(error[2])
        near = distance <= goal["replan_base_error"] and abs(turn) <= np.radians(
            goal["replan_heading_error_deg"]
        )
        rospy.loginfo(
            "[DoorOpening] base %.0f mm, %.1f deg off the planned grasp: %s",
            1000.0 * distance,
            np.degrees(turn),
            "following the plan" if near else "planning again",
        )
        return near

    def _from_door_angle(self, trajectory):
        return from_angle(
            trajectory, self._door_angle - self._config["tracking"]["catch_up_margin"]
        )

    def _run_segment(self, scene, trajectory, grasped):
        """A preempt cancels the running segment; a tracker that never answers fails it after the tracker's own
        timeout plus a margin."""
        client = self._segment_client
        if not client.wait_for_server(rospy.Duration(5.0)):
            raise OpeningFailed("no execute_door_segment server")
        self._robot.check_localization()
        client.send_goal(
            segment_goal(
                self._goal, scene, trajectory, grasped, self._holding_action, self._free_wrench
            )
        )
        duration = len(trajectory.door_angles) * trajectory.sample_period
        deadline = (
            rospy.get_time()
            + self._config["tracking"]["timeout_factor"] * duration
            + SEGMENT_TIMEOUT_MARGIN_S
        )
        while not client.wait_for_result(rospy.Duration(0.1)):
            if self._active is not None and self._active.is_preempt_requested():
                client.cancel_goal()
                client.wait_for_result(rospy.Duration(2.0))
                raise OpeningFailed("preempted")
            if rospy.get_time() > deadline or rospy.is_shutdown():
                client.cancel_goal()
                raise SegmentFailed("door segment did not end in time")
        result = client.get_result()
        if result is None:
            raise OpeningFailed("door segment returned no result")
        self._door_angle, self._laser_decided = result.final_door_angle, False
        if not result.success and is_localization_lost(result.message):
            raise LocalizationLost(result.message)
        if not result.success:
            raise SegmentFailed("door segment failed: %s" % result.message)

    # Grasping

    def _grasp_handle(self, scene, grip, planned_q):
        """Returns the grasp offset actually achieved; a grasp too far off the planned one, or away from where the
        laser sees the door, is tried again."""
        for attempt in range(self._grasp_config["grasp_attempts"]):
            if attempt > 0:
                self._stop_if_passable()
                self._release(scene)
            self._send_other_arm_home(scene, grip.arm)
            try:
                move, approach = self._plan_grasp(scene, grip, planned_q)
            except OpeningFailed as error:
                raise SegmentFailed(str(error))
            self._stage("grasp with the %s arm" % grip.arm)
            self._grasp(grip.arm, move, approach, grasp_action(self._push, grip.face))
            on_door = self._measure_door(scene, self._offsets[grip])
            measured = measure_grasp_offset(
                scene,
                self._door_angle,
                self._base_pose(),
                gripper_pose(self._arms[grip.arm], self._robot.q(grip.arm)),
            )
            error = self._grasp_error(scene, measured, grip)
            self._log_grasp(measured, grip, planned_q)
            if on_door and error <= self._grasp_config["max_grasp_error"]:
                break
        return measured

    def _grasp_error(self, scene, measured, grip):
        """A hand the force stopped is on the handle, short of the planned grasp only along its way in."""
        error = measured[:3, 3] - self._offsets[grip][:3, 3]
        if self._approach_end.ended_by != "force":
            return np.linalg.norm(error)
        arm, base = self._arms[grip.arm], self._base_pose()
        start, end = (
            measure_grasp_offset(scene, self._door_angle, base, gripper_pose(arm, q))[:3, 3]
            for q in (self._approach[0], self._approach[-1])
        )
        lateral = np.linalg.norm(lateral_error(error, end - start))
        rospy.loginfo(
            "[DoorOpening] grasp error across the approach %.0f mm", 1000.0 * lateral
        )
        return lateral

    def _grasp(self, side, move, approach, action):
        """The hand opens on the way to the handle. The wrench is zeroed with the arm still in front of the handle; a
        hand stopped by contact further out than grasp/contact_tolerance closes all the same and the grasp checks
        that follow find it off the handle."""
        grasp = self._grasp_config
        self._robot.move_hand(side, grasp["hand_open"], wait=False)
        self._robot.move_arm(side, move[1:], self._joint_speeds(side, grasp["move_speed"]))
        self._robot.wait_for_hand(side)
        zero = self._free_wrench = self._robot.zero_wrench(side)
        end = self._robot.approach_arm(
            side,
            approach,
            self._joint_speeds(side, grasp["approach_speed"]),
            grasp["contact_tolerance"],
            zero,
        )
        self._log_approach(side, approach[-1], end)
        self._robot.move_hand(side, grasp["hand_closed"])
        self._approach, self._approach_end = list(approach), end
        self._holding, self._holding_action = side, action

    def _measure_door(self, scene, offset):
        """The laser agreeing with the hand within grasp/door_agreement_deg tells that the hand holds the handle; one
        that closed on air (a released door may drift) leaves the door where the laser sees it.
        """
        gripper = gripper_pose(self._arms[self._holding], self._robot.q(self._holding))
        hand = max(
            0.0, door_angle_from_gripper(scene, offset, self._base_pose(), gripper)
        )
        reading = self._door_reading(scene)
        tolerance = np.radians(self._grasp_config["door_agreement_deg"])
        self._door_angle = door_estimate(hand, reading, tolerance)
        self._laser_decided = self._door_angle != hand
        rospy.loginfo(
            "[DoorOpening] door at %.1f deg after grasping (hand %.1f deg, laser %s)",
            np.degrees(self._door_angle),
            np.degrees(hand),
            describe_reading(reading),
        )
        return not hand_off_door(hand, reading, tolerance)

    def _door_reading(self, scene):
        try:
            scan = rospy.wait_for_message(SCAN_TOPIC, LaserScan, timeout=1.0)
        except rospy.ROSException:
            return None
        min_angle = self._door_angle - np.radians(
            self._config["tracking"]["lost_grasp_deg"]
        )
        return self._robot.door_reading(scan, self._frame, scene, min_angle)

    # Letting go and moving the arms

    def _release(self, scene):
        side, grasp = self._holding, self._grasp_config
        self._stage("releasing the %s hand" % side)
        self._released_q = self._robot.q(side)
        self._robot.move_hand(side, grasp["hand_open"])
        self._holding = None
        around = self._surroundings(scene, side)
        back_away = retreat_displacement(
            scene, self._door_angle, around.base_pose, grasp["release_drop"]
        )
        paths = release_paths(
            self._arms[side],
            self._robot.q(side),
            self._approach,
            back_away,
            self._holding_action,
            grasp,
        )
        self._approach = None
        for name, path in paths:
            if path is not None and path_leaves_clear(
                self._arms[side], around, path, self._move_collision
            ):
                self._robot.move_arm(
                    side, path[1:], self._joint_speeds(side, grasp["approach_speed"])
                )
                return
            rospy.loginfo(
                "[DoorOpening] release by %s: %s",
                name,
                "no path" if path is None else "collides",
            )
        self._log_arm_state(side)
        raise OpeningFailed("cannot move the %s hand off the handle" % side)

    def _park(self, scene, side):
        """Released drives keep the arm folded home, where the base can turn next to a wide open leaf. The released
        hand backs away from the leaf first, or folds with only the arm links kept clear of the door.
        """
        home = self._home[side]
        if (
            np.max(np.abs(self._robot.q(side) - home))
            <= self._grasp_config["home_tolerance"]
        ):
            return
        back_away = self._back_away_path(scene, side)
        try:
            if back_away is None:
                self._move_arm_to(scene, side, [home], GRASP)
            else:
                self._move_arm_to(scene, side, [home], FREE, back_away)
        except SegmentFailed:
            if back_away is not None:
                speeds = self._joint_speeds(side, self._grasp_config["approach_speed"])
                self._robot.move_arm(side, back_away[1:], speeds)
            self._step_away_from_leaf(scene, side)
            self._move_arm_to(scene, side, [home], GRASP)

    def _back_away_path(self, scene, side):
        grasp = self._grasp_config
        displacement = retreat_displacement(
            scene, self._door_angle, self._base_pose(), grasp["standoff"]
        )
        path = line_path(
            self._arms[side],
            self._robot.q(side),
            displacement,
            grasp["approach_points"],
            grasp["ik_iterations"],
        )
        if path is None or not self._path_clear(scene, side, path, GRASP):
            return None
        return path

    def _step_away_from_leaf(self, scene, side):
        """A base pressed against the leaf drives straight away from it, forwards or backwards, so the arm can fold.
        Next to the frame the way away from the leaf can run into the wall: shorter steps and the other way come next.
        Only a step after which the arm folds home clear counts, and none is needed when it folds where it stands.
        """
        base = self._base_pose()
        if self._fold_clear(scene, side, base):
            return
        heading = np.array([np.cos(base[2]), np.sin(base[2])])
        away = np.copysign(
            self._grasp_config["standoff"],
            heading @ toward_robot(scene, self._door_angle, base[:2]),
        )
        for distance in (away, away / 2.0, -away / 2.0, -away):
            drive = self._clear_straight_drive(scene, side, base, distance)
            if drive is not None and self._fold_clear(scene, side, drive.base_poses[-1]):
                self._stage("stepping the base away from the leaf")
                self._run_segment(scene, drive, False)
                return
        raise SegmentFailed("the %s arm is caught against the leaf" % side)

    def _fold_clear(self, scene, side, base_pose):
        """Whether the arm folds home on a straight joint path with the base at base_pose."""
        path = joint_path(
            self._robot.q(side), self._home[side], self._grasp_config["joint_step"]
        )
        around = self._surroundings(scene, side)._replace(base_pose=base_pose)
        return arm_path_clear(self._arms[side], around, path, GRASP, self._move_collision)

    def _clear_straight_drive(self, scene, side, base, distance):
        q = self._robot.q(side)
        drive = straight_drive(
            base,
            distance,
            self._door_angle,
            q,
            side,
            self._config["planner"]["linear_speed"],
        )
        clear = base_path_leaves_clear(
            self._arms[side],
            self._surroundings(scene, side),
            q,
            drive.base_poses,
            self._move_collision,
        )
        return drive if clear else None

    def _send_other_arm_home(self, scene, side):
        other = other_side(side)
        if (
            np.max(np.abs(self._robot.q(other) - self._home[other]))
            > self._grasp_config["home_tolerance"]
        ):
            self._stage("moving the %s arm home" % other)
            self._move_arm_to(scene, other, [self._home[other]])

    def _move_arm_to(self, scene, side, q_goals, mode=FREE, lead_in=None):
        """To the first of q_goals reachable straight in joint space or folded in through home, after the lead_in path
        at approach speed when one is given."""
        grasp = self._grasp_config
        lead_in = [self._robot.q(side)] if lead_in is None else list(lead_in)
        q, step = lead_in[-1], grasp["joint_step"]
        lead_speeds = [self._joint_speeds(side, grasp["approach_speed"])] * (
            len(lead_in) - 1
        )
        for q_goal in q_goals:
            for path in (
                joint_path(q, q_goal, step),
                via_path([q, self._home[side], q_goal], step),
            ):
                if self._path_clear(scene, side, path, mode):
                    move_speeds = [self._joint_speeds(side, grasp["move_speed"])] * (
                        len(path) - 1
                    )
                    self._robot.move_arm(
                        side,
                        lead_in[1:] + list(path[1:]),
                        np.array(lead_speeds + move_speeds),
                    )
                    return
        self._log_arm_state(side)
        raise SegmentFailed("no collision-free path to park the %s arm" % side)

    def _joint_speeds(self, side, speed):
        """speed for every joint, but no joint faster than grasp/max_joint_speed_fraction of its limit."""
        limits = velocity_limits(self._arms[side])
        return np.minimum(speed, self._grasp_config["max_joint_speed_fraction"] * limits)

    def _path_clear(self, scene, side, path, mode):
        """A hand leaving the handle keeps the same margin as the hand that came to it."""
        collision = self._move_collision if mode == GRASP else self._collision
        return arm_path_clear(
            self._arms[side], self._surroundings(scene, side), path, mode, collision
        )

    # Small helpers

    def _surroundings(self, scene, side):
        """What the arm on side moves among: the door, the costmap and the other arm where it is now."""
        other = other_side(side)
        other_spheres = link_spheres(self._arms[other], self._robot.q(other))
        return Surroundings(
            scene, self._door_angle, self._base_pose(), self._grid, other_spheres
        )

    def _base_pose(self):
        return self._robot.base_pose(self._frame, timeout=2.0)

    def _costmap(self):
        """The latest costmap; one in another frame than the door would put obstacles in the wrong place. The robot's
        own footprint is free."""
        costmap = self._config["costmap"]
        try:
            msg = rospy.wait_for_message(
                costmap["topic"], OccupancyGrid, timeout=costmap["timeout_s"]
            )
        except rospy.ROSException:
            rospy.logwarn(
                "[DoorOpening] no costmap on %s, planning with the door scene only",
                costmap["topic"],
            )
            return None
        if msg.header.frame_id.lstrip("/") != self._frame.lstrip("/"):
            rospy.logwarn(
                "[DoorOpening] costmap in %s, door in %s: planning with the door scene only",
                msg.header.frame_id,
                self._frame,
            )
            return None
        grid, cleared = occupancy_to_grid(
            msg,
            costmap["occupied_threshold"],
            self._base_pose()[:2],
            self._collision.base_radius + self._collision.margin,
        )
        if cleared:
            rospy.loginfo(
                "[DoorOpening] costmap: %d occupied cells under the robot taken as free",
                cleared,
            )
        return grid

    def _log_grasp(self, measured, grip, planned_q):
        offset = self._offsets[grip]
        error = measured[:3, 3] - offset[:3, 3]
        turn = np.degrees(
            np.arccos(
                np.clip(
                    (np.trace(offset[:3, :3].T @ measured[:3, :3]) - 1.0) / 2.0,
                    -1.0,
                    1.0,
                )
            )
        )
        rospy.loginfo(
            "[DoorOpening] hand %.0f mm from the planned grasp, %s mm in the handle frame, turned %.1f deg;"
            " base %s q %s (planned %s)",
            1000.0 * np.linalg.norm(error),
            np.round(1000.0 * error).astype(int),
            turn,
            rounded(self._base_pose()),
            rounded(self._robot.q(grip.arm)),
            rounded(planned_q),
        )

    def _log_approach(self, side, q_end, end):
        q = self._robot.q(side)
        arm = self._arms[side]
        short_mm = 1000.0 * np.linalg.norm(
            gripper_pose(arm, q)[:3, 3] - gripper_pose(arm, q_end)[:3, 3]
        )
        short_rad = np.max(np.abs(q - q_end))
        early = short_rad > self._grasp_config["contact_tolerance"]
        rospy.loginfo(
            "[DoorOpening] %s approach ended by %s: peak force %.1f N, %.3f rad / %.0f mm short of the handle%s",
            side,
            end.ended_by,
            end.peak_force,
            short_rad,
            short_mm,
            ", beyond contact_tolerance" if early else "",
        )

    def _log_arm_state(self, side):
        rospy.logwarn(
            "[DoorOpening] %s arm stuck at q %s, other arm q %s, base %s, door %.1f deg",
            side,
            rounded(self._robot.q(side)),
            rounded(self._robot.q(other_side(side))),
            rounded(self._base_pose()),
            np.degrees(self._door_angle),
        )


def main():
    rospy.init_node("door_opening_server")
    DoorOpeningServer()
    rospy.spin()


if __name__ == "__main__":
    main()
