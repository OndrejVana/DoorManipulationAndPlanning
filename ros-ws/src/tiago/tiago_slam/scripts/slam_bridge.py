#!/usr/bin/env python3
"""Hands the SLAM pose to the global TF as map -> odom, on top of the robot's own wheel odometry."""

import collections

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage

WHEEL_HISTORY_S = 2.0


def yaw_of(q):
    return np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def pose2d(translation, rotation):
    return np.array([translation.x, translation.y, yaw_of(rotation)])


def compose(a, b):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array(
        [a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]]
    )


def inverse(a):
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])


def interpolate(history, stamp):
    """Pose at stamp from (stamp, pose) pairs in time order, or None outside them."""
    times = [t for t, _ in history]
    index = int(np.searchsorted(times, stamp))
    if index == 0 or index == len(history):
        return None
    (t0, p0), (t1, p1) = history[index - 1], history[index]
    fraction = (stamp - t0) / (t1 - t0)
    turn = np.arctan2(np.sin(p1[2] - p0[2]), np.cos(p1[2] - p0[2]))
    return np.array(
        [
            p0[0] + fraction * (p1[0] - p0[0]),
            p0[1] + fraction * (p1[1] - p0[1]),
            p0[2] + fraction * turn,
        ]
    )


class SlamBridge(object):
    def __init__(self):
        self._map = rospy.get_param("~map_frame", "map")
        self._odom = rospy.get_param("~odom_frame", "odom")
        self._base = rospy.get_param("~base_frame", "base_footprint")
        self._tf_delay = rospy.get_param("~tf_delay", 0.05)
        self._map_from_slam_odom = None
        self._map_from_odom = None
        self._wheel = collections.deque()
        self._tf = rospy.Publisher("/tf", TFMessage, queue_size=10)
        self._pose = rospy.Publisher("slam_pose", PoseStamped, queue_size=10)
        rospy.Subscriber("wheel_odom", Odometry, self._on_wheel, queue_size=50)
        rospy.Subscriber("slam_tf", TFMessage, self._on_slam_tf, queue_size=100)
        rospy.Timer(rospy.Duration(1.0 / rospy.get_param("~rate", 30.0)), self._publish)

    def _on_wheel(self, msg):
        self._wheel.append(
            (
                msg.header.stamp.to_sec(),
                pose2d(msg.pose.pose.position, msg.pose.pose.orientation),
            )
        )
        while self._wheel and self._wheel[-1][0] - self._wheel[0][0] > WHEEL_HISTORY_S:
            self._wheel.popleft()

    def _on_slam_tf(self, msg):
        for transform in msg.transforms:
            parent, child = transform.header.frame_id.lstrip(
                "/"
            ), transform.child_frame_id.lstrip("/")
            pose = pose2d(transform.transform.translation, transform.transform.rotation)
            if parent == self._map and child == self._odom:
                self._map_from_slam_odom = pose
            elif (
                parent == self._odom
                and child == self._base
                and self._map_from_slam_odom is not None
            ):
                self._update(
                    transform.header.stamp, compose(self._map_from_slam_odom, pose)
                )

    def _update(self, stamp, map_from_base):
        wheel = interpolate(self._wheel, stamp.to_sec())
        if wheel is None:
            return
        self._map_from_odom = compose(map_from_base, inverse(wheel))
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = stamp, self._map
        pose.pose.position.x, pose.pose.position.y = map_from_base[:2]
        pose.pose.orientation.z, pose.pose.orientation.w = np.sin(
            map_from_base[2] / 2.0
        ), np.cos(map_from_base[2] / 2.0)
        self._pose.publish(pose)

    def _publish(self, _):
        """Stamped slightly ahead, as rtabmap does, so lookups at the newest sensor data find it."""
        if self._map_from_odom is None:
            return
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now() + rospy.Duration(self._tf_delay)
        transform.header.frame_id, transform.child_frame_id = self._map, self._odom
        transform.transform.translation.x, transform.transform.translation.y = (
            self._map_from_odom[:2]
        )
        transform.transform.rotation.z = np.sin(self._map_from_odom[2] / 2.0)
        transform.transform.rotation.w = np.cos(self._map_from_odom[2] / 2.0)
        self._tf.publish(TFMessage([transform]))


def main():
    rospy.init_node("slam_bridge")
    SlamBridge()
    rospy.spin()


if __name__ == "__main__":
    main()
