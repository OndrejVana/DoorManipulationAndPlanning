#!/usr/bin/env python3
"""Base IMU yaw rate for the SLAM's EKF: sign corrected and gyro bias removed."""

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

STILL_SPEED = 0.005  # m/s and rad/s of the wheel odometry below which the base counts as standing still


class ImuFix(object):
    def __init__(self):
        self._sign = rospy.get_param("~yaw_rate_sign", -1.0)
        self._bias_samples = rospy.get_param("~bias_samples", 50)
        self._variance = rospy.get_param("~yaw_rate_variance", 4e-4)
        self._bias_sum, self._bias_count = 0.0, 0
        self._still = False
        self._pub = rospy.Publisher("imu_out", Imu, queue_size=10)
        rospy.Subscriber("wheel_odom", Odometry, self._on_odom, queue_size=1)
        rospy.Subscriber("imu_in", Imu, self._on_imu, queue_size=10)

    def _on_odom(self, msg):
        twist = msg.twist.twist
        self._still = (
            abs(twist.linear.x) < STILL_SPEED and abs(twist.angular.z) < STILL_SPEED
        )

    def _bias(self):
        return self._bias_sum / self._bias_count if self._bias_count else 0.0

    def _on_imu(self, msg):
        if self._still and self._bias_count < self._bias_samples:
            self._bias_sum += msg.angular_velocity.z
            self._bias_count += 1
            if self._bias_count == self._bias_samples:
                rospy.loginfo("[imu_fix] gyro yaw bias %.5f rad/s", self._bias())
        msg.angular_velocity.z = self._sign * (msg.angular_velocity.z - self._bias())
        covariance = list(msg.angular_velocity_covariance)
        covariance[8] = self._variance
        msg.angular_velocity_covariance = covariance
        self._pub.publish(msg)


def main():
    rospy.init_node("imu_fix")
    ImuFix()
    rospy.spin()


if __name__ == "__main__":
    main()
