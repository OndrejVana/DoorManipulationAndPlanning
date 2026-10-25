#!/usr/bin/env python3
"""Action server executing one grasped door segment with closed-loop whole-body tracking."""

import actionlib
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tiago_door_planning.door_scene import DoorDimensions, scene_from_door_poses
from tiago_door_planning.geometry import matrix_to_pose2d, pose2d_to_matrix
from tiago_door_planning.kinematics import (
    arm_joint_names,
    clamp_to_limits,
    gripper_jacobian,
    gripper_pose,
    joint_limits,
    load_arm_model,
    velocity_limits,
)
from tiago_door_planning.msg import (
    ExecuteDoorSegmentAction,
    ExecuteDoorSegmentFeedback,
    ExecuteDoorSegmentResult,
)
from tiago_door_planning.ros_conversions import (
    joint_position_map,
    joint_positions,
    pose_msg_to_matrix,
    transform_msg_to_matrix,
)
from tiago_door_planning.tracking import (
    ABORT,
    DONE,
    JointLimits,
    Measurement,
    TrackingGains,
    measure_grasp_offset,
    tracking_step,
)
from tiago_door_planning.trajectory import DoorTrajectory, last_progress, reference_at


def tracking_gains(config, arm):
    names = [name for name in TrackingGains._fields if name != "joint_limits"]
    lower, upper = joint_limits(arm)
    limits = JointLimits(
        lower, upper, velocity_limits(arm) * config["joint_velocity_fraction"]
    )
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
    command.linear.x = linear
    command.angular.z = angular
    return command


class DoorTrackingServer(object):
    def __init__(self):
        self._config = rospy.get_param("~")
        self._frames = self._config["frames"]
        self._tracking = self._config["tracking"]
        self._urdf = rospy.get_param("/robot_description")
        self._joint_state = rospy.wait_for_message("/joint_states", JointState)
        self._tf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf)
        rospy.Subscriber(
            "/joint_states", JointState, self._on_joint_state, queue_size=1
        )
        self._cmd_vel = rospy.Publisher(
            "/mobile_base_controller/cmd_vel", Twist, queue_size=1
        )
        self._arm_publishers = {
            side: rospy.Publisher(
                "/arm_%s_controller/command" % side, JointTrajectory, queue_size=1
            )
            for side in ("left", "right")
        }
        self._server = actionlib.SimpleActionServer(
            "execute_door_segment",
            ExecuteDoorSegmentAction,
            self._execute,
            auto_start=False,
        )
        self._server.start()

    def _on_joint_state(self, msg):
        if "arm_left_1_joint" in msg.name:
            self._joint_state = msg

    def _lookup(self, target_frame, source_frame):
        stamped = self._tf.lookup_transform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0)
        )
        return transform_msg_to_matrix(stamped.transform)

    def _base_pose(self):
        return matrix_to_pose2d(
            self._lookup(self._frames["odom"], self._frames["base"])
        )

    def _scene_in_odom(self, goal):
        odom_from_map = self._lookup(
            self._frames["odom"], goal.hinge_pose.header.frame_id
        )
        hinge = odom_from_map @ pose_msg_to_matrix(goal.hinge_pose.pose)
        handle = odom_from_map @ pose_msg_to_matrix(goal.handle_pose.pose)
        dimensions = DoorDimensions(**self._config["door"])
        return (
            scene_from_door_poses(
                hinge, handle, goal.hinge_side, goal.push, dimensions
            ),
            odom_from_map,
        )

    def _trajectory_in_odom(self, goal, odom_from_map):
        poses_in_map = np.column_stack([goal.base_x, goal.base_y, goal.base_yaw])
        poses = np.array(
            [
                matrix_to_pose2d(odom_from_map @ pose2d_to_matrix(p))
                for p in poses_in_map
            ]
        )
        return DoorTrajectory(
            poses,
            np.array(goal.door_angles),
            goal.sample_period,
            goal.arm,
            np.array(goal.elbow_hint),
        )

    def _load_arm(self, side):
        arm_config = self._config["arm"]
        return load_arm_model(
            self._urdf,
            side,
            arm_config["gripper_frame"],
            arm_config["sphere_links"],
            arm_config["sphere_radius"],
            arm_config["sphere_spacing"],
            joint_position_map(self._joint_state),
        )

    def _measure(self, arm):
        q = joint_positions(self._joint_state, arm_joint_names(arm.side))
        return Measurement(
            self._base_pose(), q, gripper_pose(arm, q), gripper_jacobian(arm, q)
        )

    def _stop(self, arm, q):
        self._cmd_vel.publish(twist(0.0, 0.0))
        self._arm_publishers[arm.side].publish(
            arm_command(arm, q, np.zeros(len(q)), 0.5)
        )

    def _publish(self, arm, measurement, command):
        self._cmd_vel.publish(twist(command.linear, command.angular))
        lookahead = self._tracking["command_lookahead_s"]
        self._arm_publishers[arm.side].publish(
            arm_command(arm, measurement.q, command.joint_velocities, lookahead)
        )

    def _deadline(self, trajectory):
        duration = last_progress(trajectory) * trajectory.sample_period
        return rospy.Time.now() + rospy.Duration(
            duration * self._tracking["timeout_factor"] + 5.0
        )

    def _feedback(self, trajectory, command):
        feedback = ExecuteDoorSegmentFeedback()
        feedback.progress = command.progress / last_progress(trajectory)
        feedback.gripper_error = command.gripper_error
        self._server.publish_feedback(feedback)

    def _track(self, scene, trajectory, arm, grasp_offset):
        gains = tracking_gains(self._tracking, arm)
        dt = 1.0 / self._tracking["rate_hz"]
        rate = rospy.Rate(self._tracking["rate_hz"])
        deadline = self._deadline(trajectory)
        progress = 0.0
        while not rospy.is_shutdown():
            measurement = self._measure(arm)
            if self._server.is_preempt_requested():
                return measurement, progress, "preempted"
            if rospy.Time.now() > deadline:
                return measurement, progress, "timeout"
            command = tracking_step(
                trajectory, progress, scene, grasp_offset, measurement, gains, dt
            )
            if command.status == ABORT:
                return (
                    measurement,
                    progress,
                    "gripper error %.3f m exceeded limit" % command.gripper_error,
                )
            self._publish(arm, measurement, command)
            self._feedback(trajectory, command)
            progress = command.progress
            if command.status == DONE:
                return measurement, progress, ""
            rate.sleep()
        return self._measure(arm), progress, "shutdown"

    def _execute(self, goal):
        scene, odom_from_map = self._scene_in_odom(goal)
        trajectory = self._trajectory_in_odom(goal, odom_from_map)
        arm = self._load_arm(goal.arm)
        start = self._measure(arm)
        grasp_offset = measure_grasp_offset(
            scene, trajectory.door_angles[0], start.base_pose, start.gripper_in_base
        )
        measurement, progress, failure = self._track(
            scene, trajectory, arm, grasp_offset
        )
        self._stop(arm, measurement.q)
        self._finish(trajectory, progress, failure)

    def _finish(self, trajectory, progress, failure):
        result = ExecuteDoorSegmentResult()
        result.success = not failure
        result.final_door_angle = reference_at(trajectory, progress).door_angle
        result.message = failure or "segment completed"
        rospy.loginfo(
            "[DoorTracking] %s (door angle %.1f deg)",
            result.message,
            np.degrees(result.final_door_angle),
        )
        if failure == "preempted":
            self._server.set_preempted(result)
        elif failure:
            self._server.set_aborted(result)
        else:
            self._server.set_succeeded(result)


def main():
    rospy.init_node("door_tracking_server")
    DoorTrackingServer()
    rospy.spin()


if __name__ == "__main__":
    main()
