#!/usr/bin/env python3
"""Copies the robot's TF into the SLAM's own TF topics, without the transforms the SLAM publishes itself."""

import rospy
from tf2_msgs.msg import TFMessage


class TfFilter(object):
    def __init__(self):
        self._odom = rospy.get_param("~odom_frame", "odom")
        self._base = rospy.get_param("~base_frame", "base_footprint")
        self._tf = rospy.Publisher("tf_out", TFMessage, queue_size=100)
        self._tf_static = rospy.Publisher(
            "tf_static_out", TFMessage, queue_size=10, latch=True
        )
        self._static = {}
        rospy.Subscriber("tf_in", TFMessage, self._on_tf, queue_size=100)
        rospy.Subscriber("tf_static_in", TFMessage, self._on_tf_static, queue_size=10)

    def _kept(self, transforms):
        return [
            t
            for t in transforms
            if t.child_frame_id.lstrip("/") != self._odom
            and not (
                t.header.frame_id.lstrip("/") == self._odom
                and t.child_frame_id.lstrip("/") == self._base
            )
        ]

    def _on_tf(self, msg):
        kept = self._kept(msg.transforms)
        if kept:
            self._tf.publish(TFMessage(kept))

    def _on_tf_static(self, msg):
        """Each static publisher latches its own message; the copy latches all of them in one."""
        for transform in self._kept(msg.transforms):
            self._static[transform.child_frame_id] = transform
        self._tf_static.publish(TFMessage(list(self._static.values())))


def main():
    rospy.init_node("tf_filter")
    TfFilter()
    rospy.spin()


if __name__ == "__main__":
    main()
