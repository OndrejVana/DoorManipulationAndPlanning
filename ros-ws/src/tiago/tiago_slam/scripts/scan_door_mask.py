#!/usr/bin/env python3
"""Removes door leaf and robot artifacts from the scan."""

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PolygonStamped
from sensor_msgs.msg import LaserScan


def inside_polygon(points, polygon):
    """(N,) bools, points (N, 2) inside polygon (M, 2), by counting crossings of a ray along +x."""
    inside = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0)):
        crosses = (y0 > y) != (y1 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < x_cross)
    return inside


def masked_ranges(ranges, angles, laser_to_polygon, polygon):
    """ranges with the returns inside polygon set to inf; laser_to_polygon is the 3x3 planar laser pose."""
    finite = np.isfinite(ranges)
    r, a = ranges[finite], angles[finite]
    points = np.column_stack([r * np.cos(a), r * np.sin(a), np.ones(len(r))])
    hit = np.zeros(len(ranges), dtype=bool)
    hit[finite] = inside_polygon((points @ laser_to_polygon.T)[:, :2], polygon)
    out = np.array(ranges, dtype=float)
    out[hit] = np.inf
    return out


def planar_pose(transform):
    rotation, translation = transform.rotation, transform.translation
    yaw = 2.0 * np.arctan2(rotation.z, rotation.w)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, translation.x], [s, c, translation.y], [0.0, 0.0, 1.0]])


class ScanDoorMask(object):
    def __init__(self):
        self._polygon = None
        self._frame = None
        self._tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf)
        self._publisher = rospy.Publisher("scan_out", LaserScan, queue_size=10)
        rospy.Subscriber("polygon", PolygonStamped, self._on_polygon, queue_size=1)
        rospy.Subscriber("scan_in", LaserScan, self._on_scan, queue_size=10)

    def _on_polygon(self, msg):
        self._polygon = np.array([[p.x, p.y] for p in msg.polygon.points])
        self._frame = msg.header.frame_id

    def _on_scan(self, msg):
        if self._polygon is not None and len(self._polygon) >= 3:
            try:
                # the latest pose: the polygon's margin is far wider than the robot moves in one TF period
                pose = planar_pose(self._tf.lookup_transform(self._frame, msg.header.frame_id, rospy.Time(0)).transform)
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
                rospy.logwarn_throttle(5.0, "[ScanDoorMask] no %s pose, scan dropped", msg.header.frame_id)
                return
            angles = msg.angle_min + msg.angle_increment * np.arange(len(msg.ranges))
            msg.ranges = masked_ranges(np.array(msg.ranges), angles, pose, self._polygon).tolist()
        self._publisher.publish(msg)


def main():
    rospy.init_node("scan_door_mask")
    ScanDoorMask()
    rospy.spin()


if __name__ == "__main__":
    main()
