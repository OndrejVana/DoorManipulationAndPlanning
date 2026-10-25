#!/usr/bin/env python3
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import JointState

from tiago_door_planning.kinematics import arm_joint_names, gripper_pose, load_arm_model
from tiago_door_planning.ros_conversions import (
    joint_position_map,
    joint_positions,
    transform_msg_to_matrix,
)

GRIPPER_FRAME = "hand_{side}_grasping_frame"
BASE_FRAME = "base_footprint"


def tf_gripper_pose(buffer, side):
    stamped = buffer.lookup_transform(
        BASE_FRAME, GRIPPER_FRAME.format(side=side), rospy.Time(0), rospy.Duration(2.0)
    )
    return transform_msg_to_matrix(stamped.transform)


def fk_gripper_pose(urdf, joint_state, side):
    arm = load_arm_model(
        urdf, side, GRIPPER_FRAME, [], 0.07, 0.1, joint_position_map(joint_state)
    )
    return gripper_pose(arm, joint_positions(joint_state, arm_joint_names(side)))


def report(side, fk, tf):
    position_error = np.linalg.norm(fk[:3, 3] - tf[:3, 3])
    rotation_error = np.arccos(
        np.clip((np.trace(fk[:3, :3].T @ tf[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    )
    rospy.loginfo(
        "%s arm: position error %.2f mm, rotation error %.3f deg",
        side,
        1000.0 * position_error,
        np.degrees(rotation_error),
    )


def main():
    rospy.init_node("check_kinematics")
    buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(buffer)
    urdf = rospy.get_param("/robot_description")
    joint_state = rospy.wait_for_message("/joint_states", JointState, timeout=10.0)
    for side in ("left", "right"):
        report(
            side,
            fk_gripper_pose(urdf, joint_state, side),
            tf_gripper_pose(buffer, side),
        )


if __name__ == "__main__":
    main()
