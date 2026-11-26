#!/usr/bin/env python3
"""Sim-only stand-in for PAL's wrist F/T correction node: publishes /wrist_{side}_ft/corrected, the wrench applied on
the hand by the environment with the hand weight removed, in wrist_{side}_ft_link, torque about its origin.

It reads the wrist_{side}_ft_joint wrench Gazebo publishes every physics step (see spawn_without_safety_box.py). The
wrist chatters at the physics rate, so a single step's wrench swings by ~100 N; the mean over average_steps steps
does not.
"""

import random

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import WrenchStamped

BASE_FRAME = "base_footprint"
GRAVITY = 9.81


def hand_wrench(force, torque, gravity, mass, mass_com, force_bias, torque_bias):
    """(force, torque) the environment applies on the hand, all in the sensor axes, from the joint wrench: the load
    the hand puts on the sensor, its weight plus what the environment applies on it."""
    return force - force_bias - mass * gravity, torque - torque_bias - np.cross(mass_com, gravity)


def vector_param(name):
    return np.array(rospy.get_param("~" + name), dtype=float)


def as_array(vector):
    return np.array([vector.x, vector.y, vector.z])


class CorrectedWrench(object):
    def __init__(self, side, tf_buffer):
        self._mass = rospy.get_param("~%s/mass" % side)
        self._mass_com = vector_param(side + "/mass_com")
        self._force_bias = vector_param(side + "/force_bias")
        self._torque_bias = vector_param(side + "/torque_bias")
        self._average_steps = rospy.get_param("~average_steps")
        self._drop_probability = rospy.get_param("~drop_probability")
        self._steps = []
        self._tf = tf_buffer
        self._publisher = rospy.Publisher("/wrist_%s_ft/corrected" % side, WrenchStamped, queue_size=10)
        rospy.Subscriber("/wrist_%s_ft/joint_wrench" % side, WrenchStamped, self._on_wrench, queue_size=100)

    def _gravity_in(self, frame):
        rotation = self._tf.lookup_transform(frame, BASE_FRAME, rospy.Time(0)).transform.rotation
        x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
        # gravity is -z in base_footprint; rotate it into the sensor frame
        return -GRAVITY * np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    def _on_wrench(self, msg):
        self._steps.append(np.concatenate([as_array(msg.wrench.force), as_array(msg.wrench.torque)]))
        if len(self._steps) < self._average_steps:
            return
        mean, self._steps = np.mean(self._steps, axis=0), []
        if random.random() < self._drop_probability:
            return
        try:
            gravity = self._gravity_in(msg.header.frame_id)
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException):
            return
        force, torque = hand_wrench(
            mean[:3], mean[3:], gravity, self._mass, self._mass_com, self._force_bias, self._torque_bias
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
    rospy.loginfo("ft_corrected_sim: publishing %d corrected wrist wrenches", len(adapters))
    rospy.spin()


if __name__ == "__main__":
    main()
