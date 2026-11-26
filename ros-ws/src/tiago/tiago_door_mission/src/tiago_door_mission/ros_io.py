"""The mission's io on ROS: perception topics, the door servers, move_base, play_motion and the head."""

import threading

import actionlib
import numpy as np
import rospy
import tf2_ros
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import PointHeadAction, PointHeadGoal
from geometry_msgs.msg import Point, Point32, PolygonStamped, Pose, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import OccupancyGrid, Odometry
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
from std_msgs.msg import Bool, Int32, String

from tiago_door_planning.model.door_scene import scene_from_door_poses, swept_polygon
from tiago_door_planning.model.geometry import rotation_to_quaternion
from tiago_door_planning.msg import OpenDoorAction, OpenDoorGoal, PassDoorAction, PassDoorGoal
from tiago_door_planning.ros.config import door_dimensions, localization_params
from tiago_door_planning.ros.ros_conversions import pose_msg_to_matrix
from tiago_door_planning.sensing.localization import LocalizationWatch, is_localization_lost
from tiago_door_planning.srv import (
    MeasureDoor,
    MeasureDoorRequest,
    PlanApproach,
    PlanApproachRequest,
    TrackDoorLeaf,
    TrackDoorLeafRequest,
)

from .door_lock import DoorSample

BASE_FRAME = "base_footprint"
SERVER_WAIT_S = 2.0
TUCK_ATTEMPTS = 12
TUCK_RETRY_S = 5.0
HINGE_TOPIC = "/door/hinge_pose_map"
HANDLE_TOPIC = "/door/handle_pose_map"
TAG_TOPIC = "/door/tag_pose_map"
GOAL_STATES = {
    value: name for name, value in vars(GoalStatus).items() if name.isupper() and isinstance(value, int)
}


def pose_stamped(frame, pose):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = pose[:3, 3]
    o = msg.pose.orientation
    o.x, o.y, o.z, o.w = rotation_to_quaternion(pose[:3, :3])
    return msg


def fill_door(request, door):
    """Sets the door fields shared by every door goal and request."""
    request.hinge_pose = pose_stamped(door.frame, door.hinge)
    request.handle_pose = pose_stamped(door.frame, door.handle)
    request.hinge_side = door.hinge_side
    request.push = door.push
    return request


class DoorSampler(object):
    """Pairs each hinge pose with the handle pose of the same detection, tagged with the tag's semantics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._samples = None
        self._hinge = None
        self._meta = {"hinge_side": None, "interaction": None, "tag_id": None}
        rospy.Subscriber(HINGE_TOPIC, PoseStamped, self._on_hinge, queue_size=10)
        rospy.Subscriber(HANDLE_TOPIC, PoseStamped, self._on_handle, queue_size=10)
        rospy.Subscriber("/door/hinge_side", String, self._on_meta, "hinge_side", queue_size=10)
        rospy.Subscriber("/door/interaction", String, self._on_meta, "interaction", queue_size=10)
        rospy.Subscriber("/door/tag_id", Int32, self._on_meta, "tag_id", queue_size=10)
        rospy.Subscriber(TAG_TOPIC, PoseStamped, self._on_tag, queue_size=10)
        self.last_tag = None

    def _on_meta(self, msg, key):
        self._meta[key] = msg.data

    def _on_hinge(self, msg):
        self._hinge = msg

    def _on_tag(self, msg):
        self.last_tag = msg

    def _on_handle(self, msg):
        hinge, meta = self._hinge, dict(self._meta)
        if hinge is None or hinge.header.stamp != msg.header.stamp or None in meta.values():
            return
        sample = DoorSample(
            pose_msg_to_matrix(hinge.pose),
            pose_msg_to_matrix(msg.pose),
            meta["hinge_side"],
            meta["interaction"] == "push",
            int(meta["tag_id"]),
        )
        with self._lock:
            if self._samples is not None:
                self._samples.append(sample)

    def collect(self, count, skip, timeout, stop):
        """Up to count samples after the first skip, waiting at most timeout; fewer when the tag is out of view."""
        with self._lock:
            self._samples = []
        deadline = rospy.get_time() + timeout
        while rospy.get_time() < deadline and not stop.is_set() and not rospy.is_shutdown():
            with self._lock:
                if len(self._samples) >= skip + count:
                    break
            rospy.sleep(0.05)
        with self._lock:
            samples, self._samples = self._samples[skip:], None
        return samples


class RosMissionIO(object):
    def __init__(self, params):
        self._params = params
        self._dry_run = params["dry_run"]
        self.frame = params["frame"]
        self._stop = threading.Event()
        self._door_dimensions = door_dimensions(params["door_motion"])
        self._sampler = DoorSampler()
        self._tf = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf)
        self._state = rospy.Publisher("~state", String, queue_size=1, latch=True)
        self._mask_polygon = rospy.Publisher(params["mask"]["polygon_topic"], PolygonStamped, queue_size=1, latch=True)
        self._mask_enabled = rospy.Publisher(params["mask"]["enabled_topic"], Bool, queue_size=1, latch=True)
        self._open = actionlib.SimpleActionClient("open_door", OpenDoorAction)
        self._pass = actionlib.SimpleActionClient("pass_door", PassDoorAction)
        self._move_base = actionlib.SimpleActionClient(params["move_base_action"], MoveBaseAction)
        self._play_motion = actionlib.SimpleActionClient("/play_motion", PlayMotionAction)
        self._head = actionlib.SimpleActionClient("/head_controller/point_head_action", PointHeadAction)
        self._plan_approach = rospy.ServiceProxy("plan_approach", PlanApproach)
        self._measure_door = rospy.ServiceProxy("measure_door", MeasureDoor)
        self._track_leaf = rospy.ServiceProxy("track_door_leaf", TrackDoorLeaf)
        # Goals that move the robot end when the SLAM loses track (door_motion.yaml localization)
        self._localization = None
        localization = localization_params(params["door_motion"])
        if localization is not None and not self._dry_run:
            self._localization = LocalizationWatch(localization, rospy.get_time())
            rospy.Subscriber(localization.topic, Odometry, self._on_localization, queue_size=10)

    def _on_localization(self, msg):
        self._localization.update(rospy.get_time(), msg.pose.covariance[0])

    def _localization_lost(self):
        return None if self._localization is None else self._localization.lost(rospy.get_time())

    # Run control

    def request_stop(self):
        self._stop.set()

    def clear_stop(self):
        self._stop.clear()

    def report(self, state, message):
        self._state.publish(String(data=state if not message else "%s: %s" % (state, message)))
        if message:
            rospy.loginfo("[DoorMission] %s: %s", state, message)
        else:
            rospy.loginfo("[DoorMission] %s", state)

    # io used by the mission

    def check_system(self):
        """Names of what the mission needs and cannot reach; a dry run needs nothing that moves the robot."""
        missing = []
        clients = [("open_door", self._open), ("pass_door", self._pass)]
        if not self._dry_run:
            clients.append((self._params["move_base_action"], self._move_base))
            if self._params["tuck_arms"]:
                clients.append(("/play_motion", self._play_motion))
            if self._params["head"]["enabled"]:
                clients.append(("point_head_action", self._head))
        for name, client in clients:
            if not client.wait_for_server(rospy.Duration(SERVER_WAIT_S)):
                missing.append(name)
        services = [("plan_approach", self._plan_approach), ("measure_door", self._measure_door)]
        if self._params["mask"]["slam_leaf"]:
            services.append(("track_door_leaf", self._track_leaf))
        for name, proxy in services:
            try:
                proxy.wait_for_service(SERVER_WAIT_S)
            except rospy.ROSException:
                missing.append(name)
        try:
            self._tf.lookup_transform(self.frame, BASE_FRAME, rospy.Time(0), rospy.Duration(SERVER_WAIT_S))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            missing.append("TF %s -> %s" % (self.frame, BASE_FRAME))
        try:
            rospy.wait_for_message(self._params["costmap_topic"], OccupancyGrid, timeout=self._params["costmap_wait_s"])
        except rospy.ROSException:
            missing.append(self._params["costmap_topic"])
        lost = self._localization_lost()
        if lost is not None:
            missing.append(lost)
        return missing

    def tuck_arms(self):
        """play_motion rejects a motion while another one plays (the sim tucks the arms itself at start-up)."""
        goal = PlayMotionGoal()
        goal.motion_name = self._params["tuck_motion"]
        goal.skip_planning = False
        for _ in range(TUCK_ATTEMPTS):
            ok, message = self._run(self._play_motion, goal, self._params["tuck_timeout_s"])
            if ok or self._play_motion.get_state() != GoalStatus.REJECTED or self._stop.is_set():
                break
            rospy.logwarn("[DoorMission] play_motion busy (%s), trying again", message)
            rospy.sleep(TUCK_RETRY_S)
        return ok, message

    def look_at(self, door):
        """Points the camera at the tag where it was last seen, else at the locked handle; without either it stays put.
        The tag, not the handle: close by, the tag above an aimed-at handle leaves the image."""
        head = self._params["head"]
        if self._dry_run or not head["enabled"]:
            return
        if self._sampler.last_tag is not None:
            msg = self._sampler.last_tag
            frame = msg.header.frame_id
            target = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        elif door is not None:
            frame, target = door.frame, door.handle[:3, 3]
        else:
            return
        goal = PointHeadGoal()
        goal.target.header.frame_id = frame
        goal.target.point.x, goal.target.point.y, goal.target.point.z = target
        goal.pointing_frame = head["pointing_frame"]
        goal.pointing_axis.z = 1.0
        goal.min_duration = rospy.Duration(head["min_duration_s"])
        goal.max_velocity = head["max_velocity"]
        ok, message = self._run(self._head, goal, head["timeout_s"])
        if not ok:
            rospy.logwarn("[DoorMission] pointing the head failed: %s", message)
        rospy.sleep(head["settle_s"])

    def door_samples(self):
        lock = self._params["lock"]
        return self._sampler.collect(lock["samples"], lock["skip_samples"], lock["timeout_s"], self._stop)

    def measure_door(self, door):
        try:
            response = self._measure_door(fill_door(MeasureDoorRequest(), door))
        except rospy.ServiceException as error:
            return False, 0.0, str(error)
        rospy.loginfo("[DoorMission] laser: %s", response.message)
        return response.seen, response.angle, response.message

    def set_mask(self, door):
        if door is not None:
            mask = self._params["mask"]
            scene = scene_from_door_poses(door.hinge, door.handle, door.hinge_side, door.push, self._door_dimensions)
            polygon = PolygonStamped()
            polygon.header.frame_id = door.frame
            polygon.header.stamp = rospy.Time.now()
            polygon.polygon.points = [
                Point32(x, y, 0.0)
                for x, y in swept_polygon(scene, 0.0, np.radians(mask["max_angle_deg"]), mask["margin"])
            ]
            self._mask_polygon.publish(polygon)
        self._mask_enabled.publish(Bool(data=door is not None))
        if self._params["mask"]["slam_leaf"]:
            self._mask_leaf(door)
        self._wait_for_costmap_after(rospy.Time.now() + rospy.Duration(self._params["mask"]["settle_s"]))

    def _mask_leaf(self, door):
        """door_leaf_tracker hides the leaf from the SLAM scans until the mask is turned off; the mission goes on
        without it."""
        request = TrackDoorLeafRequest(user="mission", enable=door is not None)
        if door is not None:
            fill_door(request, door)
        try:
            self._track_leaf(request)
        except rospy.ServiceException as error:
            rospy.logwarn("[DoorMission] leaf mask failed: %s", error)

    def _wait_for_costmap_after(self, stamp):
        """The costmap applies the mask on its next update and publishes it at its own rate; a plan read from a
        costmap published before then still has the leaf in it."""
        deadline = rospy.get_time() + self._params["costmap_wait_s"] + (stamp - rospy.Time.now()).to_sec()
        while not rospy.is_shutdown() and rospy.get_time() < deadline:
            try:
                msg = rospy.wait_for_message(
                    self._params["costmap_topic"], OccupancyGrid, timeout=self._params["costmap_wait_s"]
                )
            except rospy.ROSException:
                break
            if msg.header.stamp >= stamp:
                return
        rospy.logwarn("[DoorMission] no costmap on %s after the mask changed", self._params["costmap_topic"])

    def plan_approach(self, door):
        try:
            response = self._plan_approach(fill_door(PlanApproachRequest(), door))
        except rospy.ServiceException as error:
            return [], str(error)
        return list(response.poses), response.message

    def robot_pose(self):
        t = self._tf.lookup_transform(self.frame, BASE_FRAME, rospy.Time(0), rospy.Duration(SERVER_WAIT_S)).transform
        return pose_msg_to_matrix(Pose(Point(t.translation.x, t.translation.y, 0.0), t.rotation))

    def navigate(self, pose):
        goal = MoveBaseGoal()
        goal.target_pose = pose_stamped(self.frame, pose) if isinstance(pose, np.ndarray) else pose
        goal.target_pose.header.stamp = rospy.Time.now()
        return self._run(self._move_base, goal, self._params["nav_timeout_s"])

    def open_door(self, door, plan_only):
        goal = fill_door(OpenDoorGoal(), door)
        goal.plan_only, goal.release_at_end = plan_only, True
        ok, message = self._run(self._open, goal, self._params["open_timeout_s"])
        result = self._open.get_result()
        if result is None:
            return False, 0.0, message
        if not ok and is_localization_lost(message):
            return False, result.final_door_angle, message
        return ok and result.success, result.final_door_angle, result.message

    def pass_door(self, door, angle):
        goal = fill_door(PassDoorGoal(), door)
        goal.door_angle = angle
        ok, message = self._run(self._pass, goal, self._params["pass_timeout_s"])
        result = self._pass.get_result()
        if result is None or (not ok and is_localization_lost(message)):
            return False, message
        return ok and result.success, result.message

    def stop_all(self):
        """Cancels every goal the mission may have running."""
        for client in (self._open, self._pass, self._move_base, self._play_motion):
            if client.gh is not None and client.get_state() in (GoalStatus.ACTIVE, GoalStatus.PENDING):
                client.cancel_goal()

    def _run(self, client, goal, timeout):
        """(ok, message) of one action goal; a stop request, a lost localization or the timeout cancels it."""
        if not client.wait_for_server(rospy.Duration(SERVER_WAIT_S)):
            return False, "no action server"
        lost = self._localization_lost()
        if lost is not None:
            return False, lost
        client.send_goal(goal)
        deadline = rospy.get_time() + timeout
        while not client.wait_for_result(rospy.Duration(0.2)):
            if self._stop.is_set() or rospy.is_shutdown():
                client.cancel_goal()
                client.wait_for_result(rospy.Duration(5.0))
                return False, "stopped"
            if rospy.get_time() > deadline:
                client.cancel_goal()
                client.wait_for_result(rospy.Duration(5.0))
                return False, "timed out after %.0f s" % timeout
            lost = self._localization_lost()
            if lost is not None:
                client.cancel_goal()
                client.wait_for_result(rospy.Duration(5.0))
                return False, lost
        state = client.get_state()
        if state != GoalStatus.SUCCEEDED:
            return False, "ended %s: %s" % (GOAL_STATES.get(state, state), client.get_goal_status_text())
        return True, ""
