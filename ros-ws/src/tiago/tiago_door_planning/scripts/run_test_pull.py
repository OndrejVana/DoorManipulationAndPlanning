#!/usr/bin/env python3

import actionlib
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from tiago_door_planning.door_scene import (
    DoorDimensions,
    handle_pose,
    scene_from_door_poses,
)
from tiago_door_planning.geometry import matrix_to_pose2d
from tiago_door_planning.kinematics import arm_joint_names
from tiago_door_planning.msg import ExecuteDoorSegmentAction, ExecuteDoorSegmentGoal
from tiago_door_planning.ros_conversions import (
    joint_positions,
    pose_msg_to_matrix,
    transform_msg_to_matrix,
)

DIMENSIONS = DoorDimensions(
    width=0.9, thickness=0.04, height=2.0, wall_thickness=0.15, wall_length=1.0
)


def straight_backup_poses(scene, start_pose, door_angles):
    heading = np.array([np.cos(start_pose[2]), np.sin(start_pose[2])])
    handle_start = handle_pose(scene, 0.0)[:2, 3]
    shifts = [
        (handle_pose(scene, angle)[:2, 3] - handle_start) @ heading
        for angle in door_angles
    ]
    return np.array(
        [[*(start_pose[:2] + shift * heading), start_pose[2]] for shift in shifts]
    )


def arm_for_hinge(hinge_side):
    return "right" if hinge_side == "left" else "left"


def base_pose_in(frame):
    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer)
    stamped = buffer.lookup_transform(
        frame, "base_footprint", rospy.Time(0), rospy.Duration(5.0)
    )
    return matrix_to_pose2d(transform_msg_to_matrix(stamped.transform))


def build_goal(hinge, handle, hinge_side, push, base_pose, elbow_hint):
    scene = scene_from_door_poses(
        pose_msg_to_matrix(hinge.pose),
        pose_msg_to_matrix(handle.pose),
        hinge_side,
        push,
        DIMENSIONS,
    )
    duration = rospy.get_param("~duration_s", 10.0)
    sample_period = rospy.get_param("~sample_period", 0.1)
    final_angle = np.radians(rospy.get_param("~final_angle_deg", 25.0))
    door_angles = np.linspace(
        0.0, final_angle, int(round(duration / sample_period)) + 1
    )
    poses = straight_backup_poses(scene, base_pose, door_angles)

    goal = ExecuteDoorSegmentGoal()
    goal.hinge_pose, goal.handle_pose = hinge, handle
    goal.hinge_side, goal.push, goal.arm = hinge_side, push, arm_for_hinge(hinge_side)
    goal.base_x, goal.base_y, goal.base_yaw = (
        poses[:, 0].tolist(),
        poses[:, 1].tolist(),
        poses[:, 2].tolist(),
    )
    goal.door_angles, goal.sample_period = door_angles.tolist(), sample_period
    goal.elbow_hint = list(elbow_hint)
    return goal


def main():
    rospy.init_node("run_test_pull")
    hinge = rospy.wait_for_message("/door/hinge_pose_map", PoseStamped, timeout=10.0)
    handle = rospy.wait_for_message("/door/handle_pose_map", PoseStamped, timeout=10.0)
    hinge_side = rospy.wait_for_message("/door/hinge_side", String, timeout=10.0).data
    joint_state = rospy.wait_for_message("/joint_states", JointState, timeout=10.0)
    arm = arm_for_hinge(hinge_side)
    elbow_hint = joint_positions(joint_state, arm_joint_names(arm))

    goal = build_goal(
        hinge,
        handle,
        hinge_side,
        rospy.get_param("~push", False),
        base_pose_in(hinge.header.frame_id),
        elbow_hint,
    )
    client = actionlib.SimpleActionClient(
        "execute_door_segment", ExecuteDoorSegmentAction
    )
    client.wait_for_server()
    client.send_goal(goal)
    client.wait_for_result()
    result = client.get_result()
    rospy.loginfo(
        "success=%s door=%.1f deg: %s",
        result.success,
        np.degrees(result.final_door_angle),
        result.message,
    )


if __name__ == "__main__":
    main()
