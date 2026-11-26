"""The door mission as a SMACH state machine over an io object, so it runs against ROS or test fakes alike."""

import collections

import numpy as np
import smach

from .door_lock import lock_door

SUCCEEDED, FAILED, PREEMPTED = "succeeded", "failed", "preempted"

MissionParams = collections.namedtuple(
    "MissionParams",
    "dry_run tuck_arms lock_far lock_near lock_attempts view_distances view_max_distance view_max_angle "
    "closed_tolerance_far closed_tolerance_near approach_attempts open_attempts",
)

KEYS = ["door", "approach_poses", "open_attempts", "door_angle", "message"]


def _face_normal(door):
    """Horizontal normal out of the handle's face (handle z), toward the side the robot stands on."""
    normal = door.handle[:2, 2]
    return normal / np.linalg.norm(normal)


def view_poses(door, distances):
    """Base poses at the distances in front of the handle, facing it."""
    normal = _face_normal(door)
    yaw = np.arctan2(-normal[1], -normal[0])
    poses = []
    for distance in distances:
        pose = np.eye(4)
        pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        pose[:2, 3] = door.handle[:2, 3] + distance * normal
        poses.append(pose)
    return poses


def in_view_range(door, robot_pose, max_distance, max_angle):
    """Whether the robot already sees the tag from where its pose is reliable: close by, not too far off its normal."""
    offset = robot_pose[:2, 3] - door.handle[:2, 3]
    distance = np.linalg.norm(offset)
    if distance > max_distance:
        return False
    return (
        np.arccos(np.clip(offset @ _face_normal(door) / max(distance, 1e-9), -1.0, 1.0))
        <= max_angle
    )


class Step(smach.State):
    """One mission step: fn(userdata) returns the outcome; a preempt ends the step as preempted."""

    def __init__(self, name, fn, outcomes, io):
        smach.State.__init__(self, outcomes=list(outcomes) + [PREEMPTED], io_keys=KEYS)
        self._name, self._fn, self._io = name, fn, io

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return PREEMPTED
        self._io.report(self._name, "")
        outcome = self._fn(userdata)
        if self.preempt_requested():
            self.service_preempt()
            return PREEMPTED
        return outcome


class Stop(smach.State):
    """Stops everything; runs to the end even while a preempt is pending."""

    def __init__(self, name, fn, io):
        smach.State.__init__(self, outcomes=["stopped"], io_keys=KEYS)
        self._name, self._fn, self._io = name, fn, io

    def execute(self, userdata):
        self._io.report(self._name, userdata.message)
        return self._fn(userdata)


class Mission(object):
    """The steps; each returns an outcome and leaves the reason of a failure in userdata.message."""

    def __init__(self, io, params):
        self.io, self.params = io, params

    def _fail(self, userdata, message):
        userdata.message = message
        return "failed"

    def check_system(self, userdata):
        missing = self.io.check_system()
        if missing:
            return self._fail(userdata, "not ready: %s" % ", ".join(missing))
        return "ok"

    def tuck_arms(self, userdata):
        if self.params.dry_run or not self.params.tuck_arms:
            return "ok"
        ok, message = self.io.tuck_arms()
        return (
            "ok"
            if ok
            else self._fail(userdata, "tucking the arms failed: %s" % message)
        )

    def _lock_once(self, userdata, params):
        """Looks at the door (where it was last locked, if anywhere) and locks the door model."""
        self.io.look_at(userdata.door)
        door, reason = lock_door(self.io.door_samples(), params, self.io.frame)
        if door is None:
            return self._fail(userdata, "cannot lock the door: %s" % reason)
        userdata.door = door
        return "ok"

    def _lock(self, userdata, params, state):
        """A few tries while the samples disagree; a tag out of view ends it at once."""
        for _ in range(self.params.lock_attempts):
            if self._lock_once(userdata, params) == "ok":
                return "ok"
            self.io.report(state, userdata.message)
            if "not seen" in userdata.message:
                break
        return "failed"

    def lock(self, userdata):
        """From afar the tag's orientation can be off by tens of degrees (the planar pose ambiguity, often in every
        sample alike); this lock only places the viewing pose."""
        return self._lock(userdata, self.params.lock_far, "LOCK_DOOR")

    def go_to_view(self, userdata):
        """Drives in front of the tag, where its pose is reliable, unless the robot is already there."""
        if self.params.dry_run or in_view_range(
            userdata.door,
            self.io.robot_pose(),
            self.params.view_max_distance,
            self.params.view_max_angle,
        ):
            return "ok"
        failures = []
        for pose in view_poses(userdata.door, self.params.view_distances):
            ok, message = self.io.navigate(pose)
            if ok:
                return "ok"
            failures.append(message)
        return self._fail(
            userdata,
            "cannot reach a pose to view the door from: %s" % "; ".join(failures),
        )

    def lock_view(self, userdata):
        """The lock everything after plans on: from close by, to the near spreads."""
        return self._lock(userdata, self.params.lock_near, "LOCK_VIEW")

    def refine(self, userdata):
        """The map drifts while the robot drives, so the door is locked again from close by, to the near spreads. Only
        a tag out of view keeps the earlier lock."""
        previous = userdata.door
        if self._lock(userdata, self.params.lock_near, "REFINE_DOOR") == "ok":
            self.io.set_mask(userdata.door)  # the masks follow the refined door
            return "ok"
        if "not seen" in userdata.message:
            self.io.report("REFINE_DOOR", "tag out of view; keeping the earlier lock")
            userdata.door = previous
            return "ok"
        return "failed"

    def _closed(self, userdata, require_seen, tolerance):
        seen, angle, message = self.io.measure_door(userdata.door)
        userdata.door_angle = angle if seen else 0.0
        if not seen:
            if require_seen:
                return self._fail(
                    userdata,
                    "the laser does not see the leaf (%s); cannot tell it is closed"
                    % message,
                )
            return "closed"
        if angle > tolerance:
            return self._fail(
                userdata,
                "the door is open at %.0f deg; close it and start again"
                % np.degrees(angle),
            )
        return "closed"

    def check_closed_far(self, userdata):
        """The robot still stands back from the door, where the leaf may be only partly in the laser's view; an unseen
        leaf passes and the check near the door decides."""
        return self._closed(userdata, False, self.params.closed_tolerance_far)

    def check_closed_near(self, userdata):
        return self._closed(userdata, True, self.params.closed_tolerance_near)

    def mask_on(self, userdata):
        self.io.set_mask(userdata.door)
        return "ok"

    def plan_approach(self, userdata):
        poses, message = self.io.plan_approach(userdata.door)
        if not poses:
            return self._fail(userdata, "no approach pose: %s" % message)
        userdata.approach_poses = list(poses)[: self.params.approach_attempts]
        return "ok"

    def navigate(self, userdata):
        """Tries the approach poses nearest first."""
        if self.params.dry_run:
            return "ok"
        failures = []
        for pose in userdata.approach_poses:
            ok, message = self.io.navigate(pose)
            if ok:
                return "ok"
            failures.append(message)
        return self._fail(userdata, "navigation failed: %s" % "; ".join(failures))

    def open_door(self, userdata):
        """A failed opening is tried again from a fresh approach while attempts last and the door is still closed. A door
        the opening itself left open stops the mission: from the doorway the laser can take the wall for a closed
        leaf."""
        userdata.open_attempts += 1
        ok, angle, message = self.io.open_door(userdata.door, self.params.dry_run)
        if ok:
            userdata.door_angle = angle
            return "dry_run" if self.params.dry_run else "opened"
        userdata.message = "opening failed: %s" % message
        if angle > self.params.closed_tolerance_near:
            userdata.message += (
                "; the door is left open at %.0f deg, close it and start again"
                % np.degrees(angle)
            )
            return "failed"
        if userdata.open_attempts < self.params.open_attempts:
            self.io.report("OPEN_DOOR", "%s; trying again" % userdata.message)
            return "retry"
        return "failed"

    def pass_door(self, userdata):
        ok, message = self.io.pass_door(userdata.door, userdata.door_angle)
        return "ok" if ok else self._fail(userdata, "passing failed: %s" % message)

    def mask_off(self, userdata):
        self.io.set_mask(None)
        return "ok"

    def stop(self, userdata):
        self.io.stop_all()
        self.io.set_mask(None)
        return "stopped"


def build_mission(io, params):
    """The state machine; its userdata.message holds the reason when it ends failed."""
    mission = Mission(io, params)
    machine = smach.StateMachine(outcomes=[SUCCEEDED, FAILED, PREEMPTED])
    machine.userdata.door = None
    machine.userdata.approach_poses = []
    machine.userdata.open_attempts = 0
    machine.userdata.door_angle = 0.0
    machine.userdata.message = ""

    def add(name, fn, transitions):
        transitions = dict(transitions)
        transitions.setdefault("failed", "SAFE_STOP")
        transitions.setdefault(PREEMPTED, "ABORT")
        smach.StateMachine.add(
            name,
            Step(name, fn, transitions.keys() - {PREEMPTED}, io),
            transitions=transitions,
        )

    with machine:
        add("CHECK_SYSTEM", mission.check_system, {"ok": "TUCK_ARMS"})
        add("TUCK_ARMS", mission.tuck_arms, {"ok": "LOCK_DOOR"})
        add("LOCK_DOOR", mission.lock, {"ok": "GO_TO_VIEW"})
        add("GO_TO_VIEW", mission.go_to_view, {"ok": "LOCK_VIEW"})
        add("LOCK_VIEW", mission.lock_view, {"ok": "CHECK_CLOSED"})
        add("CHECK_CLOSED", mission.check_closed_far, {"closed": "MASK_ON"})
        add("MASK_ON", mission.mask_on, {"ok": "PLAN_APPROACH"})
        add("PLAN_APPROACH", mission.plan_approach, {"ok": "NAVIGATE"})
        add("NAVIGATE", mission.navigate, {"ok": "REFINE_DOOR"})
        add("REFINE_DOOR", mission.refine, {"ok": "CHECK_CLOSED_NEAR"})
        add("CHECK_CLOSED_NEAR", mission.check_closed_near, {"closed": "OPEN_DOOR"})
        add(
            "OPEN_DOOR",
            mission.open_door,
            {"opened": "PASS_DOOR", "dry_run": "MASK_OFF", "retry": "REOBSERVE"},
        )
        add("REOBSERVE", mission.refine, {"ok": "RECHECK_CLOSED"})
        add("RECHECK_CLOSED", mission.check_closed_near, {"closed": "MASK_ON"})
        add("PASS_DOOR", mission.pass_door, {"ok": "MASK_OFF"})
        add("MASK_OFF", mission.mask_off, {"ok": SUCCEEDED})
        smach.StateMachine.add(
            "SAFE_STOP",
            Stop("SAFE_STOP", mission.stop, io),
            transitions={"stopped": FAILED},
        )
        smach.StateMachine.add(
            "ABORT", Stop("ABORT", mission.stop, io), transitions={"stopped": PREEMPTED}
        )
    return machine
