import collections

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
from geometry_msgs.msg import WrenchStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from ..sensing.door_sensing import scan_door_reading
from ..sensing.localization import LocalizationWatch
from ..sensing.wrench import contact_start, contact_update, in_contact, zero_offset
from ..model.geometry import matrix_to_pose2d
from ..motion.grasp_motion import waypoint_times
from ..model.kinematics import arm_joint_names, load_arm_model
from .config import force_params, localization_params, scan_params
from .ros_conversions import (
    joint_position_map,
    joint_positions,
    transform_msg_to_matrix,
)

SIDES = ("right", "left")
ODOM_FRAME = "odom"
BASE_FRAME = "base_footprint"
SCAN_TOPIC = "/scan_raw"


ApproachEnd = collections.namedtuple("ApproachEnd", "ended_by peak_force")


class OpeningFailed(Exception):
    pass


class SegmentFailed(OpeningFailed):
    """A door segment or an arm move did not finish; the robot is still in a known state to plan again from."""


class LocalizationLost(OpeningFailed):
    """The SLAM lost track of the robot: the robot holds still where it is, since every plan and every collision check
    from here on would use a pose that is wrong; no replanning, no letting go, no parking."""


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


def wrench_vector(msg):
    force, torque = msg.wrench.force, msg.wrench.torque
    return np.array([force.x, force.y, force.z, torque.x, torque.y, torque.z])


def send_and_wait(client, goal, duration, check=None):
    """check() runs while the goal runs; an error it raises cancels the goal, which leaves the joints where they are."""
    client.wait_for_server(rospy.Duration(5.0))
    client.send_goal(goal)
    deadline = rospy.get_time() + 3.0 * duration + 5.0
    while not client.wait_for_result(rospy.Duration(0.05)):
        try:
            if check is not None:
                check()
        except OpeningFailed:
            client.cancel_goal()
            raise
        if rospy.get_time() > deadline or rospy.is_shutdown():
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
        self._force = force_params(config)
        self._wrenches = {side: collections.deque(maxlen=250) for side in SIDES}
        for side in SIDES:
            rospy.Subscriber(
                self.wrench_topic(side),
                WrenchStamped,
                self._on_wrench,
                side,
                queue_size=50,
            )
        self._localization = None
        params = localization_params(config)
        if params is not None:
            self._localization = LocalizationWatch(params, rospy.get_time())
            rospy.Subscriber(params.topic, Odometry, self._on_localization, queue_size=10)
        if with_moves:
            self._arm_clients = trajectory_clients("arm")
            self._hand_clients = trajectory_clients("hand")

    def _on_joint_state(self, msg):
        if "arm_left_1_joint" in msg.name:
            self._joint_state = msg

    def _on_wrench(self, msg, side):
        """Kept by arrival time: the robot's header stamps run up to half a second off the laptop clock."""
        self._wrenches[side].append((rospy.get_time(), wrench_vector(msg)))

    def _on_localization(self, msg):
        self._localization.update(rospy.get_time(), msg.pose.covariance[0])

    def localization_lost(self):
        """Why the SLAM pose cannot be trusted now, or None (also when the watch is switched off)."""
        if self._localization is None:
            return None
        return self._localization.lost(rospy.get_time())

    def check_localization(self):
        reason = self.localization_lost()
        if reason is not None:
            raise LocalizationLost(reason)

    def wrench_topic(self, side):
        return self._force.topic.format(side=side)

    def wrench_samples(self, side, since):
        """(arrival time, wrench) received after since."""
        return [sample for sample in list(self._wrenches[side]) if sample[0] > since]

    def require_wrench(self, sides):
        """The opening relies on the wrist force/torque sensors, so it does not start without them."""
        since = rospy.get_time()
        deadline = since + self._force.wait_timeout_s
        missing = list(sides)
        while missing and rospy.get_time() < deadline and not rospy.is_shutdown():
            rospy.sleep(0.1)
            missing = [side for side in missing if not self.wrench_samples(side, since)]
        if missing:
            raise OpeningFailed(
                "no wrench on %s within %.1f s; the opening needs the wrist force/torque sensors"
                % (", ".join(self.wrench_topic(side) for side in missing), self._force.wait_timeout_s)
            )

    def zero_wrench(self, side):
        """The residual the sensor reads with the hand touching nothing, over the last zero_s."""
        rospy.sleep(self._force.zero_s)
        since = rospy.get_time() - self._force.zero_s
        return zero_offset([wrench for _, wrench in self.wrench_samples(side, since)])

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
            arm.get("body_spheres", ()),
            arm.get("body_check_from"),
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
        client = self._arm_clients[side]
        goal, duration = self._arm_goal(side, qs, speeds)
        self.check_localization()
        send_and_wait(client, goal, duration, self.check_localization)
        self._arm_result(side, client, qs[-1], contact_tolerance)

    def approach_arm(self, side, qs, speeds, contact_tolerance, zero):
        """Moves onto the handle until the zeroed force shows contact, then cancels the move so the arm holds where it
        touched instead of pressing on to the planned end; a controller stopping first ends it as move_arm does."""
        client = self._arm_clients[side]
        goal, duration = self._arm_goal(side, qs, speeds)
        self.check_localization()
        client.wait_for_server(rospy.Duration(5.0))
        client.send_goal(goal)
        state, seen = contact_start(), rospy.get_time()
        deadline = seen + 3.0 * duration + 5.0
        while not client.wait_for_result(rospy.Duration(0.02)):
            for stamp, wrench in self.wrench_samples(side, seen):
                state, seen = contact_update(state, stamp, wrench, zero, self._force), stamp
                if in_contact(state, self._force):
                    client.cancel_goal()
                    client.wait_for_result(rospy.Duration(1.0))
                    return ApproachEnd("force", state.peak)
            if rospy.get_time() > deadline:
                client.cancel_goal()
                raise OpeningFailed("trajectory timed out")
            try:
                self.check_localization()
            except LocalizationLost:
                client.cancel_goal()
                raise
        stopped = self._arm_result(side, client, qs[-1], contact_tolerance)
        return ApproachEnd("controller" if stopped else "planned end", state.peak)

    def _arm_goal(self, side, qs, speeds):
        times = waypoint_times(
            [self.q(side)] + list(qs), speeds, self._config["grasp"]["min_move_s"]
        )
        return trajectory_goal(arm_joint_names(side), qs, times), times[-1]

    def _arm_result(self, side, client, q_end, contact_tolerance):
        """True when the controller stopped the arm within contact_tolerance of q_end, False when it got there."""
        if client.get_state() == GoalStatus.SUCCEEDED:
            return False
        result = client.get_result()
        error_code = None if result is None else result.error_code
        short_by = np.max(np.abs(self.q(side) - q_end))
        if (
            error_code == FollowJointTrajectoryResult.GOAL_TOLERANCE_VIOLATED
            and short_by <= contact_tolerance
        ):
            rospy.loginfo(
                "[DoorOpening] %s arm stopped %.3f rad short, on the handle",
                side,
                short_by,
            )
            return True
        raise SegmentFailed(
            "arm trajectory failed (error code %s, %.3f rad short, %s)"
            % (error_code, short_by, client.get_goal_status_text())
        )

    def move_hand(self, side, positions, wait=True):
        """Fingers stop on the handle, so the hand goal is not required to succeed."""
        duration = self._config["grasp"]["hand_duration_s"]
        goal = trajectory_goal(hand_joint_names(side), [positions], [duration])
        if wait:
            send_and_wait(self._hand_clients[side], goal, duration, self.check_localization)
        else:
            self._hand_clients[side].send_goal(goal)

    def wait_for_hand(self, side):
        self._hand_clients[side].wait_for_result(
            rospy.Duration(self._config["grasp"]["hand_duration_s"] + 1.0)
        )
