#!/usr/bin/env python3
"""Spawn the robot from /robot_description without the hand safety-box collisions.

PAL wraps each Hey5 hand in an invisible box meant for MoveIt; Gazebo treats it as solid, so the
fingers could never close around a handle. Remaining arguments are passed on to spawn_model.
"""

import os
import sys
import xml.etree.ElementTree as ElementTree

import rospy

SOURCE_PARAM = "/robot_description"
TARGET_PARAM = "/robot_description_gazebo"


def without_safety_boxes(urdf):
    robot = ElementTree.fromstring(urdf)
    for link in robot.findall("link"):
        if link.get("name", "").endswith("_safety_box"):
            for collision in link.findall("collision"):
                link.remove(collision)
    return ElementTree.tostring(robot, encoding="unicode")


def main():
    rospy.init_node("spawn_without_safety_box")
    while not rospy.has_param(SOURCE_PARAM) and not rospy.is_shutdown():
        rospy.sleep(0.5)
    rospy.set_param(TARGET_PARAM, without_safety_boxes(rospy.get_param(SOURCE_PARAM)))
    arguments = rospy.myargv(sys.argv)[1:]
    os.execvp("rosrun", ["rosrun", "gazebo_ros", "spawn_model", "-urdf", "-param", TARGET_PARAM] + arguments)


if __name__ == "__main__":
    main()
