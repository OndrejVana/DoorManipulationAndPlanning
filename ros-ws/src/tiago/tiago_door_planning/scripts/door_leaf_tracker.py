#!/usr/bin/env python3
"""Tracks the door angle with the base laser while asked"""

import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import Point32, PolygonStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64

from tiago_door_planning.model.door_scene import scene_from_door_poses, swept_polygon
from tiago_door_planning.msg import ExecuteDoorSegmentActionFeedback
from tiago_door_planning.ros.config import door_dimensions, scan_params
from tiago_door_planning.ros.robot import SCAN_TOPIC
from tiago_door_planning.ros.ros_conversions import pose_msg_to_matrix, transform_msg_to_matrix
from tiago_door_planning.sensing.door_sensing import scan_points_in, tracked_door_angle
from tiago_door_planning.srv import TrackDoorLeaf, TrackDoorLeafResponse


class DoorLeafTracker(object):
    def __init__(self):
        config = rospy.get_param("~")
        self._scan_params = scan_params(config)
        self._dimensions = door_dimensions(config)
        mask = config["leaf_mask"]
        self._window = np.radians(mask["window_deg"])
        self._margin = mask["margin"]
        self._angle_margin = np.radians(mask["angle_margin_deg"])
        self._hint_max_age = rospy.Duration(mask["hint_max_age_s"])
        self._lock = threading.Lock()
        self._users = set()
        self._scene, self._frame, self._angle = None, None, None
        self._hint, self._hint_stamp = None, rospy.Time(0)
        self._tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf)
        self._polygon = rospy.Publisher("/door_leaf/polygon", PolygonStamped, queue_size=1, latch=True)
        self._angle_out = rospy.Publisher("/door_leaf/angle", Float64, queue_size=1)
        rospy.Service("track_door_leaf", TrackDoorLeaf, self._on_request)
        rospy.Subscriber(
            "/execute_door_segment/feedback", ExecuteDoorSegmentActionFeedback, self._on_feedback, queue_size=1
        )
        rospy.Subscriber(SCAN_TOPIC, LaserScan, self._on_scan, queue_size=1)
        self._publish(None, rospy.Time.now())

    def _on_request(self, request):
        with self._lock:
            if request.enable:
                self._users.add(request.user)
                self._frame = request.hinge_pose.header.frame_id
                self._scene = scene_from_door_poses(
                    pose_msg_to_matrix(request.hinge_pose.pose),
                    pose_msg_to_matrix(request.handle_pose.pose),
                    request.hinge_side,
                    request.push,
                    self._dimensions,
                )
            else:
                self._users.discard(request.user)
                if not self._users:
                    self._scene, self._angle = None, None
            users = sorted(self._users)
        if not users:
            self._publish(None, rospy.Time.now())
        message = "tracking for %s" % ", ".join(users) if users else "not tracking"
        rospy.loginfo("[DoorLeafTracker] %s %s: %s", request.user, "on" if request.enable else "off", message)
        return TrackDoorLeafResponse(message=message)

    def _on_feedback(self, msg):
        self._hint, self._hint_stamp = msg.feedback.door_angle, msg.header.stamp

    def _on_scan(self, msg):
        with self._lock:
            scene, frame, angle = self._scene, self._frame, self._angle
        if scene is None:
            return
        try:
            frame_from_laser = transform_msg_to_matrix(
                self._tf.lookup_transform(frame, msg.header.frame_id, rospy.Time(0)).transform
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            rospy.logwarn_throttle(5.0, "[DoorLeafTracker] no %s pose in %s", msg.header.frame_id, frame)
            return
        hint = self._hint if msg.header.stamp - self._hint_stamp < self._hint_max_age else None
        near = [a for a in (angle, hint) if a is not None]
        seen = tracked_door_angle(scene, scan_points_in(msg, frame_from_laser), self._scan_params, near, self._window)
        if seen is not None:
            angle = seen
        with self._lock:
            if self._scene is not scene:
                return
            self._angle = angle
        span = [a for a in (angle, hint) if a is not None]
        self._publish(None if not span else (scene, frame, min(span), max(span)), msg.header.stamp)
        if angle is not None:
            self._angle_out.publish(Float64(data=angle))

    def _publish(self, leaf, stamp):
        """The outline of the leaf swept between two angles; an empty polygon without a leaf. The angle tracked last
        stays until the users change: a door the laser no longer sees is not in the scans to mask."""
        polygon = PolygonStamped()
        polygon.header.stamp = stamp
        if leaf is not None:
            scene, frame, low, high = leaf
            polygon.header.frame_id = frame
            polygon.polygon.points = [
                Point32(x, y, 0.0)
                for x, y in swept_polygon(
                    scene, max(0.0, low - self._angle_margin), high + self._angle_margin, self._margin
                )
            ]
        self._polygon.publish(polygon)


def main():
    rospy.init_node("door_leaf_tracker")
    DoorLeafTracker()
    rospy.spin()


if __name__ == "__main__":
    main()
