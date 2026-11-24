#!/usr/bin/env python3
"""Sends the perceived door to the open_door action and logs the result; push comes from the tag (/door/interaction).
~plan_only and ~release_at_end set the goal's flags. The door mission (tiago_door_mission) does all of this and more."""

import actionlib
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from tiago_door_planning.msg import OpenDoorAction, OpenDoorGoal


def perceived_door_goal():
    goal = OpenDoorGoal()
    goal.hinge_pose = rospy.wait_for_message("/door/hinge_pose_map", PoseStamped, timeout=10.0)
    goal.handle_pose = rospy.wait_for_message("/door/handle_pose_map", PoseStamped, timeout=10.0)
    goal.hinge_side = rospy.wait_for_message("/door/hinge_side", String, timeout=10.0).data
    goal.push = rospy.wait_for_message("/door/interaction", String, timeout=10.0).data == "push"
    goal.plan_only = rospy.get_param("~plan_only", False)
    goal.release_at_end = rospy.get_param("~release_at_end", False)
    return goal


def log_stage(feedback):
    rospy.loginfo("stage: %s", feedback.stage)


def main():
    rospy.init_node("run_open_door")
    client = actionlib.SimpleActionClient("open_door", OpenDoorAction)
    client.wait_for_server()
    client.send_goal(perceived_door_goal(), feedback_cb=log_stage)
    client.wait_for_result()
    result = client.get_result()
    rospy.loginfo("success=%s door=%.1f deg: %s", result.success, np.degrees(result.final_door_angle), result.message)


if __name__ == "__main__":
    main()
