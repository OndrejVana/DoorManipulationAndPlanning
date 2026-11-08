#!/usr/bin/env python3
"""Action server opening a door: plan the grasp and the opening, grasp, then track the grasped segments"""

import collections

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
from tiago_door_planning.model.collision import FREE, GRASP, distance_grid
from tiago_door_planning.planning.door_planner import Grip, PlanningProblem
from tiago_door_planning.sensing.door_sensing import door_estimate, hand_off_door
from tiago_door_planning.model.geometry import rotation_to_quaternion, wrap_angle
from tiago_door_planning.motion.grasp_motion import (
    joint_path,
    line_path,
    retreat_displacement,
    toward_robot,
)
from tiago_door_planning.motion.grasp_motion import unrolled_offset, via_path
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
from tiago_door_planning.model.kinematics import gripper_pose, link_spheres
from tiago_door_planning.ros.config import collision_params, goal_params, planner_params
from tiago_door_planning.msg import (
    ExecuteDoorSegmentAction,
    ExecuteDoorSegmentGoal,
    OpenDoorAction,
)
from tiago_door_planning.msg import OpenDoorFeedback, OpenDoorResult
from tiago_door_planning.planning.opening_plan import plan_opening
from tiago_door_planning.planning.plan_output import plan_to_segments
from tiago_door_planning.ros.robot import (
    SCAN_TOPIC,
    SIDES,
    OpeningFailed,
    Robot,
    SegmentFailed,
)
from tiago_door_planning.ros.ros_conversions import pose_msg_to_matrix
from tiago_door_planning.motion.tracking import (
    door_angle_from_gripper,
    measure_grasp_offset,
)
from tiago_door_planning.motion.trajectory import (
    from_angle,
    split_by_sweep,
    straight_drive,
)

COSTMAP_TOPIC = "/tiago_move_base/move_base/global_costmap/costmap"

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


def occupancy_to_grid(msg, occupied_threshold):
    info = msg.info
    occupancy = np.array(msg.data, dtype=float).reshape(info.height, info.width)
    origin = [info.origin.position.x, info.origin.position.y]
    return distance_grid(occupancy, info.resolution, origin, occupied_threshold)


def segment_goal(goal, scene, trajectory, grasped):
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
    segment.grasped = grasped
    segment.base_x, segment.base_y, segment.base_yaw = trajectory.base_poses.T.tolist()
    segment.door_angles = trajectory.door_angles.tolist()
    segment.sample_period = trajectory.sample_period
    segment.arm_qs = trajectory.arm_qs.ravel().tolist()
    return segment


class DoorOpeningServer(object):
    def __init__(self):
        self._config = config = rospy.get_param("~")
        self._grasp_config = config["grasp"]
        self._collision = collision_params(config)
        self._move_collision = self._collision._replace(
            margin=config["grasp"]["move_margin"]
        )
        self._max_sweep = np.radians(config["grasp"]["max_sweep_deg"])
        self._robot = Robot(config, with_moves=True)
        self._segment_client = actionlib.SimpleActionClient(
            "execute_door_segment", ExecuteDoorSegmentAction
        )
        self._server = actionlib.SimpleActionServer(
            "open_door", OpenDoorAction, self._execute, auto_start=False
        )
        self._server.start()

    # The action

    def _execute(self, goal):
        self._goal, self._push = goal, goal.push
        self._door_angle, self._passable_angle, self._laser_decided = 0.0, np.inf, False
        self._goals = None
        self._holding, self._holding_action, self._approach = None, None, None
        self._released_q = None
        result = OpenDoorResult()
        try:
            self._open()
            result.success, result.message = True, "door opened"
        except OpeningFailed as error:
            result.success, result.message = False, str(error)
        result.final_door_angle = self._door_angle
        rospy.loginfo(
            "[DoorOpening] %s (door angle %.1f deg)",
            result.message,
            np.degrees(self._door_angle),
        )
        if result.message == "preempted":
            self._server.set_preempted(result)
        elif result.success:
            self._server.set_succeeded(result)
        else:
            self._server.set_aborted(result)

    def _stage(self, text):
        rospy.loginfo("[DoorOpening] %s", text)
        feedback = OpenDoorFeedback()
        feedback.stage = text
        self._server.publish_feedback(feedback)
        if self._server.is_preempt_requested():
            raise OpeningFailed("preempted")

    def _open(self):
        self._stage("planning")
        self._set_up_door()
        plan = self._choose_plan()
        grip = self._last_grip = segment_grip(plan.segments[0])
        self._stage("grasp 1 with the %s arm" % grip.arm)
        self._grasp(
            grip.arm, plan.move, plan.approach, grasp_action(self._push, grip.face)
        )
        self._measure_door(self._scenes["front"], self._offsets[grip])
        self._open_with_recoveries(plan.segments)

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
                if self._passable():
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
        if not self._passable():
            raise SegmentFailed(
                "the plan ended with the door at %.0f deg, short of the %.0f deg to pass"
                % (np.degrees(self._door_angle), np.degrees(self._passable_angle))
            )

    def _stop_if_passable(self):
        """Every further grasp costs a release, a drive and a regrasp, which a door open wide enough does not need."""
        if self._passable():
            raise DoorOpen()

    def _passable(self):
        """A door angle the laser fit gave against the hand must clear the passable angle by the fit's error."""
        margin = (
            np.radians(self._config["scan"]["fit_error_deg"])
            if self._laser_decided
            else 0.0
        )
        return self._door_angle - margin >= self._passable_angle

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
        self._stage("planning with the %s arm" % side)
        grip = Grip(side, "front")
        base_pose = self._base_pose()
        move, approach = self._plan_grasp(self._scenes["front"], grip, None)
        return OpeningPlan(
            move,
            approach,
            self._plan_opening(base_pose, self._offsets, grip, approach[-1]),
        )

    def _plan_opening(self, base_pose, offsets, grip, grasp_q, grasped=True):
        """Replans keep the goal angles of the first plan; the door and the costmap stay the same."""
        checks = arm_checks(
            self._config,
            self._push,
            self._scenes,
            offsets,
            self._normals,
            self._arms,
            self._home,
        )
        problem = PlanningProblem(
            self._scenes["front"],
            self._grid,
            base_pose,
            self._door_angle,
            grasp_q,
            self._home[grip.arm],
            checks,
            grip,
            grasped,
            self._collision,
        )
        started = rospy.get_time()
        steps, self._goals = plan_opening(
            problem,
            planner_params(self._config),
            goal_params(self._config),
            self._goals,
        )
        rospy.loginfo("[DoorOpening] planning took %.1f s", rospy.get_time() - started)
        if self._goals is None:
            raise OpeningFailed("door cannot be opened wide enough to pass from here")
        self._passable_angle = self._goals[0]
        if steps is None:
            raise OpeningFailed(
                "no opening plan to %.0f deg" % np.degrees(self._passable_angle)
            )
        return plan_to_segments(steps)

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
                if self._door_moved_since_release(self._scenes["front"]):
                    self._stop_if_passable()
                    segments = self._replan_released([segment] + segments)
                    continue
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

    def _door_moved_since_release(self, scene):
        """A push grasp cannot brake the door, so it swings on after the hand lets go; the laser tells where it
        stopped."""
        tolerance = np.radians(self._grasp_config["door_agreement_deg"])
        reading = self._door_reading(scene)
        angle = door_estimate(self._door_angle, reading, tolerance)
        if angle == self._door_angle:
            return False
        rospy.loginfo(
            "[DoorOpening] the door swung on to %.1f deg after the release (laser %s)",
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
        self._segment_client.wait_for_server(rospy.Duration(5.0))
        self._segment_client.send_goal(
            segment_goal(self._goal, scene, trajectory, grasped)
        )
        self._segment_client.wait_for_result()
        result = self._segment_client.get_result()
        if result is None:
            raise OpeningFailed("door segment returned no result")
        self._door_angle, self._laser_decided = result.final_door_angle, False
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
            error = np.linalg.norm(measured[:3, 3] - self._offsets[grip][:3, 3])
            self._log_grasp(measured, grip, planned_q)
            if on_door and error <= self._grasp_config["max_grasp_error"]:
                break
        return measured

    def _grasp(self, side, move, approach, action):
        """The hand opens on the way to the handle."""
        grasp = self._grasp_config
        self._robot.move_hand(side, grasp["hand_open"], wait=False)
        self._robot.move_arm(side, move[1:], grasp["move_speed"])
        self._robot.wait_for_hand(side)
        self._robot.move_arm(
            side, approach, grasp["approach_speed"], grasp["contact_tolerance"]
        )
        self._robot.move_hand(side, grasp["hand_closed"])
        self._approach = list(approach)
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
                self._robot.move_arm(side, path[1:], grasp["approach_speed"])
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
        mode = FREE if self._back_away(scene, side) else GRASP
        try:
            self._move_arm_to(scene, side, [home], mode)
        except SegmentFailed:
            self._step_away_from_leaf(scene, side)
            self._move_arm_to(scene, side, [home], GRASP)

    def _back_away(self, scene, side):
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
            return False
        self._robot.move_arm(side, path[1:], grasp["approach_speed"])
        return True

    def _step_away_from_leaf(self, scene, side):
        """A base pressed against the leaf drives straight away from it, forwards or backwards, so the arm can fold."""
        base = self._base_pose()
        heading = np.array([np.cos(base[2]), np.sin(base[2])])
        distance = np.copysign(
            self._grasp_config["standoff"],
            heading @ toward_robot(scene, self._door_angle, base[:2]),
        )
        q = self._robot.q(side)
        drive = straight_drive(
            base,
            distance,
            self._door_angle,
            q,
            side,
            self._config["planner"]["linear_speed"],
        )
        if not base_path_leaves_clear(
            self._arms[side],
            self._surroundings(scene, side),
            q,
            drive.base_poses,
            self._move_collision,
        ):
            raise SegmentFailed("the %s arm is caught against the leaf" % side)
        self._stage("stepping the base away from the leaf")
        self._run_segment(scene, drive, False)

    def _send_other_arm_home(self, scene, side):
        other = other_side(side)
        if (
            np.max(np.abs(self._robot.q(other) - self._home[other]))
            > self._grasp_config["home_tolerance"]
        ):
            self._stage("moving the %s arm home" % other)
            self._move_arm_to(scene, other, [self._home[other]])

    def _move_arm_to(self, scene, side, q_goals, mode=FREE):
        """To the first of q_goals reachable straight in joint space or folded in through home."""
        q, step = self._robot.q(side), self._grasp_config["joint_step"]
        for q_goal in q_goals:
            for path in (
                joint_path(q, q_goal, step),
                via_path([q, self._home[side], q_goal], step),
            ):
                if self._path_clear(scene, side, path, mode):
                    self._robot.move_arm(
                        side, path[1:], self._grasp_config["move_speed"]
                    )
                    return
        self._log_arm_state(side)
        raise SegmentFailed("no collision-free path to park the %s arm" % side)

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
        costmap = self._config["costmap"]
        try:
            msg = rospy.wait_for_message(
                COSTMAP_TOPIC, OccupancyGrid, timeout=costmap["timeout_s"]
            )
        except rospy.ROSException:
            rospy.logwarn(
                "[DoorOpening] no costmap on %s, planning with the door scene only",
                COSTMAP_TOPIC,
            )
            return None
        return occupancy_to_grid(msg, costmap["occupied_threshold"])

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
