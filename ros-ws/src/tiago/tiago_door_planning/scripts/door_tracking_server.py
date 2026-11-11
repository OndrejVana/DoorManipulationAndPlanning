#!/usr/bin/env python3
"""Action server executing one door segment: closed-loop whole-body tracking, or the base alone when released."""

import actionlib
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tiago_door_planning.model.door_scene import scene_from_door_poses
from tiago_door_planning.sensing.door_sensing import door_estimate, hand_off_door, unseen_leaf_bound
from tiago_door_planning.model.geometry import matrix_to_pose2d, pose2d_to_matrix
from tiago_door_planning.model.kinematics import (
    arm_joint_names,
    clamp_to_limits,
    gripper_jacobian,
    gripper_pose,
    joint_limits,
    velocity_limits,
)
from tiago_door_planning.ros.config import apply_pipeline_speed, door_dimensions
from tiago_door_planning.msg import (
    ExecuteDoorSegmentAction,
    ExecuteDoorSegmentFeedback,
    ExecuteDoorSegmentResult,
)
from tiago_door_planning.ros.robot import ODOM_FRAME, SCAN_TOPIC, Robot
from tiago_door_planning.ros.ros_conversions import pose_msg_to_matrix
from tiago_door_planning.motion.tracking import (
    ABORT,
    DONE,
    JointLimits,
    Measurement,
    TrackingGains,
    base_tracking_step,
    door_angle_from_gripper,
    integrate_joint_command,
    measure_grasp_offset,
    tracking_step,
)
from tiago_door_planning.motion.trajectory import (
    DoorTrajectory,
    last_progress,
    reference_at,
)


def tracking_gains(config, arm):
    lower, upper = joint_limits(arm)
    limits = JointLimits(
        lower, upper, velocity_limits(arm) * config["joint_velocity_fraction"]
    )
    names = [name for name in TrackingGains._fields if name != "joint_limits"]
    return TrackingGains(joint_limits=limits, **{name: config[name] for name in names})


def arm_command(arm, q, joint_velocities, lookahead):
    point = JointTrajectoryPoint()
    point.positions = list(clamp_to_limits(arm, q + joint_velocities * lookahead))
    point.velocities = list(joint_velocities)
    point.time_from_start = rospy.Duration(lookahead)
    command = JointTrajectory()
    command.joint_names = arm_joint_names(arm.side)
    command.points = [point]
    return command


def twist(linear, angular):
    command = Twist()
    command.linear.x, command.angular.z = linear, angular
    return command


def rounded(values, digits):
    return np.round(values, digits).tolist()


class DoorTrackingServer(object):
    def __init__(self):
        self._config = apply_pipeline_speed(rospy.get_param("~"))
        self._tracking = self._config["tracking"]
        self._robot = Robot(self._config)
        self._cmd_vel = rospy.Publisher(
            "/mobile_base_controller/cmd_vel", Twist, queue_size=1
        )
        self._arm_publishers = {
            side: rospy.Publisher(
                "/arm_%s_controller/command" % side, JointTrajectory, queue_size=1
            )
            for side in ("left", "right")
        }
        self._scan = None
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self._on_scan, queue_size=1)
        self._server = actionlib.SimpleActionServer(
            "execute_door_segment",
            ExecuteDoorSegmentAction,
            self._execute,
            auto_start=False,
        )
        self._server.start()

    def _on_scan(self, msg):
        self._scan = msg

    def _execute(self, goal):
        odom_from_map = self._robot.lookup(ODOM_FRAME, goal.hinge_pose.header.frame_id)
        self._scene = self._scene_in_odom(goal, odom_from_map)
        trajectory = self._trajectory_in_odom(goal, odom_from_map)
        self._min_door_angle = trajectory.door_angles[0] - self._lost_grasp_tolerance()
        self._scan_door = (None, None)
        arm = self._robot.load_arm(goal.arm)
        start = self._measure(arm)
        grasp_offset = None
        if goal.grasped:
            grasp_offset = measure_grasp_offset(
                self._scene,
                trajectory.door_angles[0],
                start.base_pose,
                start.gripper_in_base,
            )
        self._log_base_error("start", trajectory, 0.0, start)
        measurement, progress, failure = self._track(trajectory, arm, grasp_offset)
        self._log_base_error("end", trajectory, progress, measurement)
        if grasp_offset is not None:
            self._log_slip(grasp_offset, measurement)
        self._stop(arm, measurement.q, goal.grasped)
        self._finish(
            self._final_door_angle(trajectory, progress, grasp_offset, measurement),
            failure,
        )

    def _scene_in_odom(self, goal, odom_from_map):
        hinge = odom_from_map @ pose_msg_to_matrix(goal.hinge_pose.pose)
        handle = odom_from_map @ pose_msg_to_matrix(goal.handle_pose.pose)
        return scene_from_door_poses(
            hinge, handle, goal.hinge_side, goal.push, door_dimensions(self._config)
        )

    @staticmethod
    def _trajectory_in_odom(goal, odom_from_map):
        poses_in_map = np.column_stack([goal.base_x, goal.base_y, goal.base_yaw])
        poses = np.array(
            [
                matrix_to_pose2d(odom_from_map @ pose2d_to_matrix(p))
                for p in poses_in_map
            ]
        )
        arm_qs = np.array(goal.arm_qs).reshape(len(goal.door_angles), -1)
        return DoorTrajectory(
            poses, np.array(goal.door_angles), goal.sample_period, goal.arm, arm_qs
        )

    def _track(self, trajectory, arm, grasp_offset):
        """Returns (last measurement, progress, failure message or "")."""
        gains = tracking_gains(self._tracking, arm)
        dt = 1.0 / self._tracking["rate_hz"]
        rate = rospy.Rate(self._tracking["rate_hz"])
        duration = last_progress(trajectory) * trajectory.sample_period
        deadline = rospy.Time.now() + rospy.Duration(
            duration * self._tracking["timeout_factor"] + 5.0
        )
        progress, q_command = 0.0, self._measure(arm).q
        stall_progress, stall_time = progress, rospy.Time.now()
        lost_since = None
        while not rospy.is_shutdown():
            measurement, now = self._measure(arm), rospy.Time.now()
            if self._server.is_preempt_requested():
                return measurement, progress, "preempted"
            if now > deadline:
                return measurement, progress, "timeout"
            if now - stall_time > rospy.Duration(self._tracking["stall_timeout_s"]):
                if progress - stall_progress < 1.0:
                    rospy.loginfo(
                        "[DoorTracking] stalled with q %s, planned %s",
                        rounded(measurement.q, 2),
                        rounded(reference_at(trajectory, progress).arm_q, 2),
                    )
                    return (
                        measurement,
                        progress,
                        "stalled, gripper cannot follow the handle",
                    )
                stall_progress, stall_time = progress, now
            command = self._step(
                trajectory, progress, grasp_offset, measurement, gains, dt
            )
            if command.status == ABORT:
                return (
                    measurement,
                    progress,
                    "gripper error %.3f m exceeded limit" % command.gripper_error,
                )
            if grasp_offset is None or not self._off_door(command.door_angle):
                lost_since = None
            elif lost_since is None:
                lost_since = now
            elif now - lost_since > rospy.Duration(self._tracking["lost_grasp_s"]):
                return (
                    measurement,
                    progress,
                    "lost the handle: hand at %.0f deg, door at %.0f deg"
                    % (
                        np.degrees(command.door_angle),
                        np.degrees(self._door_estimate(command.door_angle)),
                    ),
                )
            q_command = integrate_joint_command(
                q_command,
                measurement.q,
                command.joint_velocities,
                dt,
                self._tracking["max_command_lead"],
            )
            self._publish(arm, q_command, command, grasp_offset is not None)
            self._feedback(trajectory, command)
            progress = command.progress
            if command.status == DONE:
                return measurement, progress, ""
            rate.sleep()
        return self._measure(arm), progress, "shutdown"

    def _step(self, trajectory, progress, grasp_offset, measurement, gains, dt):
        if grasp_offset is None:
            return base_tracking_step(trajectory, progress, measurement, gains, dt)
        reading = self._door_from_scan()
        at_least = None if reading is None else unseen_leaf_bound(reading)
        return tracking_step(
            trajectory,
            progress,
            self._scene,
            grasp_offset,
            measurement,
            gains,
            dt,
            at_least,
        )

    def _measure(self, arm):
        q = self._robot.q(arm.side)
        base_pose = self._robot.base_pose(ODOM_FRAME)
        return Measurement(base_pose, q, gripper_pose(arm, q), gripper_jacobian(arm, q))

    def _door_from_scan(self):
        """DoorReading from the latest scan, or None without a recent scan."""
        scan = self._scan
        if scan is None or rospy.Time.now() - scan.header.stamp > rospy.Duration(
            self._config["scan"]["max_age_s"]
        ):
            return None
        if self._scan_door[0] != scan.header.stamp:
            reading = self._robot.door_reading(
                scan, ODOM_FRAME, self._scene, self._min_door_angle
            )
            self._scan_door = (scan.header.stamp, reading)
        return self._scan_door[1]

    def _lost_grasp_tolerance(self):
        return np.radians(self._tracking["lost_grasp_deg"])

    def _off_door(self, hand_angle):
        return hand_off_door(
            hand_angle, self._door_from_scan(), self._lost_grasp_tolerance()
        )

    def _door_estimate(self, hand_angle):
        return door_estimate(
            hand_angle, self._door_from_scan(), self._lost_grasp_tolerance()
        )

    def _final_door_angle(self, trajectory, progress, grasp_offset, measurement):
        """Where the hand holds the door, or where the plan left a released door, checked against the laser."""
        if grasp_offset is None:
            return self._door_estimate(reference_at(trajectory, progress).door_angle)
        hand = door_angle_from_gripper(
            self._scene,
            grasp_offset,
            measurement.base_pose,
            measurement.gripper_in_base,
        )
        return self._door_estimate(hand)

    def _publish(self, arm, q_command, command, grasped):
        self._cmd_vel.publish(twist(command.linear, command.angular))
        if grasped:
            lookahead = self._tracking["command_lookahead_s"]
            self._arm_publishers[arm.side].publish(
                arm_command(arm, q_command, command.joint_velocities, lookahead)
            )

    def _stop(self, arm, q, grasped):
        self._cmd_vel.publish(twist(0.0, 0.0))
        if grasped:
            self._arm_publishers[arm.side].publish(
                arm_command(arm, q, np.zeros(len(q)), 0.5)
            )

    def _feedback(self, trajectory, command):
        feedback = ExecuteDoorSegmentFeedback()
        feedback.progress = command.progress / last_progress(trajectory)
        feedback.gripper_error, feedback.base_error = (
            command.gripper_error,
            command.base_error,
        )
        feedback.door_angle = command.door_angle
        self._server.publish_feedback(feedback)

    def _finish(self, final_door_angle, failure):
        result = ExecuteDoorSegmentResult()
        result.success, result.final_door_angle = not failure, final_door_angle
        result.message = failure or "segment completed"
        rospy.loginfo(
            "[DoorTracking] %s (door angle %.1f deg)",
            result.message,
            np.degrees(final_door_angle),
        )
        if failure == "preempted":
            self._server.set_preempted(result)
        elif failure:
            self._server.set_aborted(result)
        else:
            self._server.set_succeeded(result)

    def _log_slip(self, grasp_offset, measurement):
        """How far the hand moved on the handle during the segment, with the door angle from the laser."""
        reading = self._door_from_scan()
        if reading is None or reading.angle is None:
            return
        held = measure_grasp_offset(
            self._scene,
            reading.angle,
            measurement.base_pose,
            measurement.gripper_in_base,
        )
        rospy.loginfo(
            "[DoorTracking] hand moved %s mm on the handle (handle frame)",
            np.round(1000.0 * (held[:3, 3] - grasp_offset[:3, 3])).astype(int).tolist(),
        )

    @staticmethod
    def _log_base_error(moment, trajectory, progress, measurement):
        rospy.loginfo(
            "[DoorTracking] base at the %s: %s, reference %s",
            moment,
            rounded(measurement.base_pose, 3),
            rounded(reference_at(trajectory, progress).base_pose, 3),
        )


def main():
    rospy.init_node("door_tracking_server")
    DoorTrackingServer()
    rospy.spin()


if __name__ == "__main__":
    main()
