#!/usr/bin/env python3
"""Runs the door mission on ~start: lock the door in view, approach it, open it and drive through. ~stop preempts."""

import threading

import numpy as np
import rospy
import smach
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

from tiago_door_mission.door_lock import LockParams
from tiago_door_mission.mission import MissionParams, build_mission
from tiago_door_mission.ros_io import RosMissionIO


def lock_params(lock, spreads):
    return LockParams(
        lock["min_samples"],
        lock[spreads]["max_position_spread"],
        np.radians(lock[spreads]["max_yaw_spread_deg"]),
    )


def mission_params(params):
    lock = params["lock"]
    return MissionParams(
        dry_run=params["dry_run"],
        tuck_arms=params["tuck_arms"],
        lock_far=lock_params(lock, "far"),
        lock_near=lock_params(lock, "near"),
        lock_attempts=lock["attempts"],
        view_distances=params["view"]["distances"],
        view_max_distance=params["view"]["max_distance"],
        view_max_angle=np.radians(params["view"]["max_angle_deg"]),
        closed_tolerance_far=np.radians(params["closed_tolerance_deg"]["far"]),
        closed_tolerance_near=np.radians(params["closed_tolerance_deg"]["near"]),
        approach_attempts=params["approach_attempts"],
        open_attempts=params["open_attempts"],
    )


class DoorMissionNode(object):
    def __init__(self):
        params = rospy.get_param("~")
        self._params = mission_params(params)
        self._io = RosMissionIO(params)
        self._machine = None
        self._thread = None
        self._result = rospy.Publisher("~result", String, queue_size=1, latch=True)
        rospy.Service("~start", Trigger, self._on_start)
        rospy.Service("~stop", Trigger, self._on_stop)
        self._io.report("IDLE", "dry run" if self._params.dry_run else "")

    def _running(self):
        return self._thread is not None and self._thread.is_alive()

    def _on_start(self, _):
        if self._running():
            return TriggerResponse(False, "the mission is already running")
        self._io.clear_stop()
        self._machine = build_mission(self._io, self._params)
        self._thread = threading.Thread(target=self._run, args=(self._machine,))
        self._thread.daemon = True
        self._thread.start()
        return TriggerResponse(True, "started")

    def _on_stop(self, _):
        if not self._running():
            return TriggerResponse(False, "the mission is not running")
        self._io.request_stop()
        self._machine.request_preempt()
        return TriggerResponse(True, "stopping")

    def _run(self, machine):
        outcome = machine.execute()
        message = machine.userdata.message
        text = outcome if outcome != "failed" else "failed: %s" % message
        self._result.publish(String(data=text))
        rospy.loginfo("[DoorMission] mission %s", text)
        self._io.report("IDLE", text)


def main():
    rospy.init_node("door_mission")
    smach.set_loggers(rospy.logdebug, rospy.logwarn, rospy.logdebug, rospy.logerr)
    DoorMissionNode()
    rospy.spin()


if __name__ == "__main__":
    main()
