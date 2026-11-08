#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from tiago_door_planning.model.door_scene import (
    door_box,
    scene_from_door_poses,
    wall_boxes,
)
from tiago_door_planning.model.geometry import rotation_to_quaternion
from tiago_door_planning.msg import (
    ExecuteDoorSegmentActionFeedback,
    ExecuteDoorSegmentActionGoal,
)
from tiago_door_planning.msg import OpenDoorActionFeedback, OpenDoorActionGoal
from tiago_door_planning.ros.config import door_dimensions
from tiago_door_planning.ros.ros_conversions import pose_msg_to_matrix

FRAME = "map"
WALL = (0.55, 0.55, 0.55, 0.6)
LEAF_NOW = (0.85, 0.45, 0.15, 0.7)
LEAF_SEGMENT_END = (0.85, 0.85, 0.85, 0.25)
GRASPED_PATH = (0.15, 0.75, 0.35, 1.0)
RELEASED_PATH = (0.25, 0.45, 0.95, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)


def new_marker(namespace, index, kind, color):
    marker = Marker()
    marker.header.frame_id = FRAME
    marker.header.stamp = rospy.Time.now()
    marker.ns, marker.id, marker.type, marker.action = (
        namespace,
        index,
        kind,
        Marker.ADD,
    )
    marker.pose.orientation.w = 1.0
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    return marker


def box_marker(box, namespace, index, color):
    marker = new_marker(namespace, index, Marker.CUBE, color)
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = box.pose[
        :3, 3
    ]
    orientation = marker.pose.orientation
    orientation.x, orientation.y, orientation.z, orientation.w = rotation_to_quaternion(
        box.pose[:3, :3]
    )
    marker.scale.x, marker.scale.y, marker.scale.z = 2.0 * box.half_extents
    return marker


def path_marker(points, grasped, index):
    marker = new_marker(
        "planned_path",
        index,
        Marker.LINE_STRIP,
        GRASPED_PATH if grasped else RELEASED_PATH,
    )
    marker.points = [Point(x, y, 0.03) for x, y in points]
    marker.scale.x = 0.025
    return marker


def text_marker(text, position):
    marker = new_marker("stage", 0, Marker.TEXT_VIEW_FACING, WHITE)
    marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position
    marker.scale.z = 0.15
    marker.text = text
    return marker


class DoorPlanViz(object):
    def __init__(self):
        self.dimensions = door_dimensions(rospy.get_param("~"))
        self.publisher = rospy.Publisher(
            "door_plan_viz/markers", MarkerArray, queue_size=1, latch=True
        )
        self.scene = None
        self.paths = []
        self.segment_end_angle = None
        self.door_angle = 0.0
        self.stage = "waiting for open_door"
        rospy.Subscriber("open_door/goal", OpenDoorActionGoal, self.on_opening)
        rospy.Subscriber("open_door/feedback", OpenDoorActionFeedback, self.on_stage)
        rospy.Subscriber(
            "execute_door_segment/goal", ExecuteDoorSegmentActionGoal, self.on_segment
        )
        rospy.Subscriber(
            "execute_door_segment/feedback",
            ExecuteDoorSegmentActionFeedback,
            self.on_progress,
        )

    def on_opening(self, message):
        goal = message.goal
        hinge, handle = pose_msg_to_matrix(goal.hinge_pose.pose), pose_msg_to_matrix(
            goal.handle_pose.pose
        )
        self.scene = scene_from_door_poses(
            hinge, handle, goal.hinge_side, goal.push, self.dimensions
        )
        self.paths, self.segment_end_angle, self.door_angle = [], None, 0.0

    def on_stage(self, message):
        self.stage = message.feedback.stage

    def on_segment(self, message):
        goal = message.goal
        self.paths.append((list(zip(goal.base_x, goal.base_y)), goal.grasped))
        self.segment_end_angle = goal.door_angles[-1] if goal.door_angles else None

    def on_progress(self, message):
        self.door_angle = message.feedback.door_angle

    def markers(self):
        markers = [
            path_marker(points, grasped, i)
            for i, (points, grasped) in enumerate(self.paths)
        ]
        if self.scene is None:
            return markers + [text_marker(self.stage, (0.0, 0.0, 2.4))]
        markers += [
            box_marker(box, "walls", i, WALL)
            for i, box in enumerate(wall_boxes(self.scene))
        ]
        markers.append(
            box_marker(door_box(self.scene, self.door_angle), "leaf", 0, LEAF_NOW)
        )
        if self.segment_end_angle is not None:
            end_box = door_box(self.scene, self.segment_end_angle)
            markers.append(box_marker(end_box, "leaf_segment_end", 0, LEAF_SEGMENT_END))
        text = "%s | door %.0f deg" % (self.stage, np.degrees(self.door_angle))
        markers.append(
            text_marker(text, (self.scene.hinge[0], self.scene.hinge[1], 2.4))
        )
        return markers

    def publish(self):
        self.publisher.publish(MarkerArray(markers=self.markers()))


def main():
    rospy.init_node("door_plan_viz")
    viz = DoorPlanViz()
    rate = rospy.Rate(5.0)
    while not rospy.is_shutdown():
        viz.publish()
        rate.sleep()


if __name__ == "__main__":
    main()
