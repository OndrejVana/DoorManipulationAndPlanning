import actionlib
import numpy as np
import rospy
import tf2_ros
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    FollowJointTrajectoryResult,
)
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from ..sensing.door_sensing import scan_door_reading
from ..model.geometry import matrix_to_pose2d
from ..motion.grasp_motion import waypoint_times
from ..model.kinematics import arm_joint_names, load_arm_model
from .config import scan_params
from .ros_conversions import (
    joint_position_map,
    joint_positions,
    transform_msg_to_matrix,
)

SIDES = ("right", "left")
ODOM_FRAME = "odom"
BASE_FRAME = "base_footprint"
SCAN_TOPIC = "/scan_raw"


class OpeningFailed(Exception):
    pass


class SegmentFailed(OpeningFailed):
    """A door segment or an arm move did not finish; the robot is still in a known state to plan again from."""


def hand_joint_names(side):
    return ["hand_%s_%s_joint" % (side, name) for name in ("thumb", "index", "mrl")]


def trajectory_goal(joint_names, qs, times):
    """A waypoint reached no later than the one before it is where the arm already is, so it is left out."""
    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = joint_names
    last = 0.0
    for q, time in zip(qs, times):
        if time <= last:
            continue
        point = JointTrajectoryPoint()
        point.positions = list(q)
        point.time_from_start = rospy.Duration(time)
        goal.trajectory.points.append(point)
        last = time
    return goal


def trajectory_clients(group):
    return {
        side: actionlib.SimpleActionClient(
            "/%s_%s_controller/follow_joint_trajectory" % (group, side),
            FollowJointTrajectoryAction,
        )
        for side in SIDES
    }


def send_and_wait(client, goal, duration):
    client.wait_for_server(rospy.Duration(5.0))
    client.send_goal(goal)
    if not client.wait_for_result(rospy.Duration(3.0 * duration + 5.0)):
        client.cancel_goal()
        raise OpeningFailed("trajectory timed out")
    return client.get_state()


class Robot(object):
    def __init__(self, config, with_moves=False):
        self._config = config
        self._urdf = rospy.get_param("/robot_description")
        self._joint_state = rospy.wait_for_message("/joint_states", JointState)
        rospy.Subscriber(
            "/joint_states", JointState, self._on_joint_state, queue_size=1
        )
        self._tf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf)
        if with_moves:
            self._arm_clients = trajectory_clients("arm")
            self._hand_clients = trajectory_clients("hand")

    def _on_joint_state(self, msg):
        if "arm_left_1_joint" in msg.name:
            self._joint_state = msg

    def q(self, side):
        return joint_positions(self._joint_state, arm_joint_names(side))

    def lookup(self, target_frame, source_frame, timeout=1.0):
        stamped = self._tf.lookup_transform(
            target_frame, source_frame, rospy.Time(0), rospy.Duration(timeout)
        )
        return transform_msg_to_matrix(stamped.transform)

    def base_pose(self, frame, timeout=1.0):
        return matrix_to_pose2d(self.lookup(frame, BASE_FRAME, timeout))

    def load_arm(self, side):
        arm = self._config["arm"]
        return load_arm_model(
            self._urdf,
            side,
            arm["gripper_frame"],
            arm["sphere_links"],
            arm["sphere_radius"],
            arm["sphere_spacing"],
            joint_position_map(self._joint_state),
        )

    def door_reading(self, scan, frame, scene, min_angle):
        """DoorReading from a scan, with the scene given in frame."""
        frame_from_laser = self.lookup(frame, scan.header.frame_id)
        return scan_door_reading(
            scan, frame_from_laser, scene, scan_params(self._config), min_angle
        )

    def move_arm(self, side, qs, speeds, contact_tolerance=0.0):
        """A hand pressing on the handle stops short of the last configuration; within contact_tolerance of it the
        move counts as done."""
        times = waypoint_times(
            [self.q(side)] + list(qs), speeds, self._config["grasp"]["min_move_s"]
        )
        client = self._arm_clients[side]
        state = send_and_wait(
            client, trajectory_goal(arm_joint_names(side), qs, times), times[-1]
        )
        if state == GoalStatus.SUCCEEDED:
            return
        result = client.get_result()
        error_code = None if result is None else result.error_code
        short_by = np.max(np.abs(self.q(side) - qs[-1]))
        if (
            error_code == FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED
            and short_by <= contact_tolerance
        ):
            rospy.loginfo(
                "[DoorOpening] %s arm stopped %.3f rad short, on the handle",
                side,
                short_by,
            )
            return
        raise SegmentFailed(
            "arm trajectory failed (error code %s, %.3f rad short, %s)"
            % (error_code, short_by, client.get_goal_status_text())
        )

    def move_hand(self, side, positions, wait=True):
        """Fingers stop on the handle, so the hand goal is not required to succeed."""
        duration = self._config["grasp"]["hand_duration_s"]
        goal = trajectory_goal(hand_joint_names(side), [positions], [duration])
        if wait:
            send_and_wait(self._hand_clients[side], goal, duration)
        else:
            self._hand_clients[side].send_goal(goal)

    def wait_for_hand(self, side):
        self._hand_clients[side].wait_for_result(
            rospy.Duration(self._config["grasp"]["hand_duration_s"] + 1.0)
        )
