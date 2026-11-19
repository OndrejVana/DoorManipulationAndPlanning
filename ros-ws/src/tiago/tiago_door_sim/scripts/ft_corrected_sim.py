#!/usr/bin/env python3
"""Sim-only stand-in for PAL's wrist F/T correction node: publishes /wrist_{side}_ft/corrected, the wrench applied on
the hand by the environment with the hand weight removed, in wrist_{side}_ft_link.
"""

import random

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import WrenchStamped

BASE_FRAME = "base_footprint"
GRAVITY = 9.81


def hand_wrench(torque, gravity, mass_com, bias, lever_arm, sensor_offset):
    """(force, torque about the sensor) from the joint-7 torque, all in the sensor axes: lever_arm is the grasp point
    and sensor_offset the sensor origin, both from the joint-7 origin."""
    external = torque - np.cross(mass_com, gravity) - bias
    force = np.cross(external, lever_arm) / (lever_arm @ lever_arm)
    return force, external - np.cross(sensor_offset, force)


def vector_param(name):
    return np.array(rospy.get_param("~" + name), dtype=float)


def as_array(vector):
    return np.array([vector.x, vector.y, vector.z])


class CorrectedWrench(object):
    def __init__(self, side, tf_buffer):
        self._mass_com = vector_param(side + "/mass_com")
        self._bias = vector_param(side + "/bias")
        self._torque_rotation = vector_param(side + "/torque_rotation").reshape(3, 3)
        self._lever_arm = vector_param("lever_arm")
        self._sensor_offset = vector_param("sensor_offset")
        self._drop_probability = rospy.get_param("~drop_probability")
        self._tf = tf_buffer
        self._publisher = rospy.Publisher(
            "/wrist_%s_ft/corrected" % side, WrenchStamped, queue_size=10
        )
        rospy.Subscriber(
            "/wrist_%s_ft" % side, WrenchStamped, self._on_wrench, queue_size=10
        )

    def _gravity_in(self, frame):
        rotation = self._tf.lookup_transform(
            frame, BASE_FRAME, rospy.Time(0)
        ).transform.rotation
        x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
        # gravity is -z in base_footprint; rotate it into the sensor frame
        return -GRAVITY * np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )

    def _on_wrench(self, msg):
        if random.random() < self._drop_probability:
            return
        try:
            gravity = self._gravity_in(msg.header.frame_id)
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ):
            return
        force, torque = hand_wrench(
            self._torque_rotation @ as_array(msg.wrench.torque),
            gravity,
            self._mass_com,
            self._bias,
            self._lever_arm,
            self._sensor_offset,
        )
        out = WrenchStamped()
        out.header = msg.header
        out.wrench.force.x, out.wrench.force.y, out.wrench.force.z = force
        out.wrench.torque.x, out.wrench.torque.y, out.wrench.torque.z = torque
        self._publisher.publish(out)


def main():
    rospy.init_node("ft_corrected_sim")
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer)
    adapters = [CorrectedWrench(side, tf_buffer) for side in rospy.get_param("~sides")]
    rospy.loginfo(
        "ft_corrected_sim: publishing %d corrected wrist wrenches", len(adapters)
    )
    rospy.spin()


if __name__ == "__main__":
    main()
