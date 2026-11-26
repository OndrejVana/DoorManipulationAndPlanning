#!/usr/bin/env python3
"""Spawn the robot from /robot_description without the hand safety-box collisions and with a wrist F/T sensor.

PAL wraps each Hey5 hand in an invisible box meant for MoveIt; Gazebo treats it as solid, so the
fingers could never close around a handle. PAL's Gazebo F/T reads the arm_*_7_joint wrench, whose force is
useless; here the fixed wrist_*_ft_joint is kept as a joint of its own and its wrench, everything the hand takes,
is published on /wrist_*_ft/joint_wrench. Remaining arguments are passed on to spawn_model.
"""

import os
import sys
import xml.etree.ElementTree as ElementTree

import rospy

SOURCE_PARAM = "/robot_description"
TARGET_PARAM = "/robot_description_gazebo"
FT_JOINT = "wrist_%s_ft_joint"


def without_safety_boxes(urdf):
    robot = ElementTree.fromstring(urdf)
    for link in robot.findall("link"):
        if link.get("name", "").endswith("_safety_box"):
            for collision in link.findall("collision"):
                link.remove(collision)
    return robot


def with_ft_joints(robot):
    """Keeps each wrist F/T joint out of Gazebo's fixed-joint lumping and publishes its wrench."""
    joints = {joint.get("name") for joint in robot.findall("joint")}
    for side in ("left", "right"):
        joint = FT_JOINT % side
        if joint not in joints:
            continue
        reference = ElementTree.SubElement(robot, "gazebo", reference=joint)
        ElementTree.SubElement(reference, "preserveFixedJoint").text = "true"
        ElementTree.SubElement(reference, "provideFeedback").text = "true"
        plugin = ElementTree.SubElement(
            ElementTree.SubElement(robot, "gazebo"),
            "plugin",
            name="wrist_%s_ft_joint_wrench" % side,
            filename="libgazebo_ros_ft_sensor.so",
        )
        ElementTree.SubElement(plugin, "jointName").text = joint
        ElementTree.SubElement(plugin, "topicName").text = "/wrist_%s_ft/joint_wrench" % side
        ElementTree.SubElement(plugin, "updateRate").text = "0"  # every physics step, see ft_corrected_sim
    return robot


def main():
    rospy.init_node("spawn_without_safety_box")
    while not rospy.has_param(SOURCE_PARAM) and not rospy.is_shutdown():
        rospy.sleep(0.5)
    robot = with_ft_joints(without_safety_boxes(rospy.get_param(SOURCE_PARAM)))
    rospy.set_param(TARGET_PARAM, ElementTree.tostring(robot, encoding="unicode"))
    arguments = rospy.myargv(sys.argv)[1:]
    os.execvp("rosrun", ["rosrun", "gazebo_ros", "spawn_model", "-urdf", "-param", TARGET_PARAM] + arguments)


if __name__ == "__main__":
    main()
