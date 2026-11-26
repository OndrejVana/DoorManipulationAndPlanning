import numpy as np
import smach
import pytest

from tiago_door_mission.door_lock import DoorSample, LockParams
from tiago_door_mission.mission import FAILED, PREEMPTED, SUCCEEDED, MissionParams, build_mission, in_view_range, view_poses

smach.set_loggers(lambda *a: None, lambda *a: None, lambda *a: None, lambda *a: None)

PARAMS = MissionParams(
    dry_run=False,
    tuck_arms=True,
    lock_far=LockParams(min_samples=3, max_position_spread=0.05, max_yaw_spread=np.radians(6.0)),
    lock_near=LockParams(min_samples=3, max_position_spread=0.02, max_yaw_spread=np.radians(3.0)),
    lock_attempts=2,
    view_distances=[1.2, 1.0],
    view_max_distance=1.5,
    view_max_angle=np.radians(35.0),
    closed_tolerance_far=np.radians(10.0),
    closed_tolerance_near=np.radians(4.0),
    approach_attempts=2,
    open_attempts=2,
)


def sample():
    """Handle at (0, 0.8), its face (z) toward +y."""
    hinge, handle = np.eye(4), np.eye(4)
    handle[:3, :3] = np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    handle[:3, 3] = [0.0, 0.8, 1.0]
    return DoorSample(hinge, handle, "left", True, 1)


def robot_at(x, y):
    pose = np.eye(4)
    pose[:2, 3] = [x, y]
    return pose


class FakeIO(object):
    """Scripted answers; records the calls."""

    frame = "map"

    def __init__(self, **answers):
        self.calls = []
        self.missing = answers.get("missing", [])
        self.samples = answers.get("samples", [sample()] * 5)
        self.angles = list(answers.get("angles", [0.0] * 10))
        self.seen = answers.get("seen", True)
        self.navigations = list(answers.get("navigations", [True] * 10))
        self.openings = list(answers.get("openings", [True]))
        self.passed = answers.get("passed", True)
        self.on_open = answers.get("on_open")
        self.failed_angle = answers.get("failed_angle", 0.0)
        self.nav_message = answers.get("nav_message", "nav")
        self.open_message = answers.get("open_message", "segment failed")
        self.robot = answers.get("robot", robot_at(0.0, 2.0))

    def _call(self, name, *args):
        self.calls.append((name,) + args)

    def names(self):
        return [call[0] for call in self.calls]

    def check_system(self):
        self._call("check_system")
        return self.missing

    def tuck_arms(self):
        self._call("tuck_arms")
        return True, ""

    def look_at(self, door):
        self._call("look_at", door)

    def door_samples(self):
        self._call("door_samples")
        return self.samples

    def robot_pose(self):
        return self.robot

    def measure_door(self, door):
        self._call("measure_door")
        return self.seen, self.angles.pop(0), "reading"

    def set_mask(self, door):
        self._call("set_mask", door is not None)

    def plan_approach(self, door):
        self._call("plan_approach")
        return ["pose a", "pose b", "pose c"], "3 poses"

    def navigate(self, pose):
        self._call("navigate", pose)
        return self.navigations.pop(0), self.nav_message

    def open_door(self, door, plan_only):
        self._call("open_door", plan_only)
        if self.on_open:
            self.on_open()
        ok = self.openings.pop(0)
        return ok, np.radians(80.0) if ok else self.failed_angle, "opened" if ok else self.open_message

    def pass_door(self, door, angle):
        self._call("pass_door", angle)
        return self.passed, "passed"

    def stop_all(self):
        self._call("stop_all")

    def report(self, state, message):
        self.calls.append(("report", state, message))


def run(io, params=PARAMS):
    machine = build_mission(io, params)
    return machine.execute(), machine.userdata


def test_a_closed_door_in_view_is_opened_and_passed():
    io = FakeIO()

    outcome, _ = run(io)

    assert outcome == SUCCEEDED
    steps = [name for name in io.names() if name != "report"]
    assert steps.index("navigate") < steps.index("open_door") < steps.index("pass_door")
    assert ("set_mask", False) in io.calls
    assert ("pass_door", np.radians(80.0)) in io.calls


def test_missing_servers_stop_before_anything_moves():
    io = FakeIO(missing=["open_door"])

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "open_door" in userdata.message
    assert "navigate" not in io.names() and "tuck_arms" not in io.names()


def test_no_tag_in_view_fails_with_door_not_seen():
    io = FakeIO(samples=[])

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "not seen" in userdata.message


def test_an_open_door_is_refused():
    io = FakeIO(angles=[np.radians(20.0)])

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "close it" in userdata.message
    assert "navigate" not in io.names()


def test_the_next_approach_pose_is_tried_when_navigation_fails():
    io = FakeIO(navigations=[False, True])

    outcome, _ = run(io)

    assert outcome == SUCCEEDED
    assert [call[1] for call in io.calls if call[0] == "navigate"] == ["pose a", "pose b"]


def test_a_failed_opening_is_tried_again_from_a_new_approach():
    io = FakeIO(openings=[False, True])

    outcome, _ = run(io)

    assert outcome == SUCCEEDED
    assert io.names().count("open_door") == 2
    assert io.names().count("plan_approach") == 2


def test_a_failed_opening_leaving_the_door_open_stops():
    io = FakeIO(openings=[False, True], angles=[0.0, 0.0, np.radians(15.0)])

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "close it" in userdata.message
    assert io.names().count("open_door") == 1


def test_an_opening_that_left_the_door_open_stops_whatever_the_laser_says():
    io = FakeIO(openings=[False, True], failed_angle=np.radians(80.0))

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "left open at 80 deg" in userdata.message
    assert io.names().count("open_door") == 1
    assert "stop_all" in io.names()


def test_openings_stop_after_the_last_attempt():
    io = FakeIO(openings=[False, False])

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "segment failed" in userdata.message
    assert io.names()[-1] in ("stop_all", "set_mask", "report")
    assert "stop_all" in io.names()


def test_a_dry_run_plans_without_moving():
    io = FakeIO()

    outcome, _ = run(io, PARAMS._replace(dry_run=True))

    assert outcome == SUCCEEDED
    assert ("open_door", True) in io.calls
    assert not {"navigate", "tuck_arms", "pass_door"} & set(io.names())


def test_near_the_door_an_unseen_leaf_stops():
    io = FakeIO()
    io.seen = True
    answers = iter([True, False])
    io.measure_door = lambda door: (next(answers), 0.0, "no fit")

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "does not see the leaf" in userdata.message


def test_a_preempt_stops_the_robot():
    holder = {}
    io = FakeIO(on_open=lambda: holder["machine"].request_preempt())
    machine = build_mission(io, PARAMS)
    holder["machine"] = machine

    outcome = machine.execute()

    assert outcome == PREEMPTED
    assert "stop_all" in io.names()
    assert "pass_door" not in io.names()


def scattered():
    spread = []
    for index in range(5):
        base = sample()
        handle = base.handle.copy()
        handle[0, 3] += 0.2 * index
        spread.append(base._replace(handle=handle))
    return spread


def test_near_the_door_a_scattered_lock_is_tried_again():
    io = FakeIO()
    answers = iter([[sample()] * 5, [sample()] * 5, scattered(), [sample()] * 5])
    io.door_samples = lambda: next(answers)

    outcome, _ = run(io)

    assert outcome == SUCCEEDED


def test_near_the_door_a_lock_that_never_agrees_stops():
    io = FakeIO()
    answers = iter([[sample()] * 5, [sample()] * 5, scattered(), scattered()])
    io.door_samples = lambda: next(answers)

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "spread" in userdata.message


def test_near_the_door_an_unseen_tag_keeps_the_earlier_lock():
    io = FakeIO()
    answers = iter([[sample()] * 5, [sample()] * 5, []])
    io.door_samples = lambda: next(answers)

    outcome, _ = run(io)

    assert outcome == SUCCEEDED


def test_from_afar_the_robot_drives_in_front_of_the_tag_and_locks_there():
    io = FakeIO(robot=robot_at(1.0, 4.0))

    outcome, _ = run(io)

    assert outcome == SUCCEEDED
    view = [call[1] for call in io.calls if call[0] == "navigate"][0]
    assert view[:2, 3] == pytest.approx([0.0, 2.0])
    assert view[:2, 0] == pytest.approx([0.0, -1.0])
    states = [call[1] for call in io.calls if call[0] == "report"]
    assert states.index("GO_TO_VIEW") < states.index("LOCK_VIEW") < states.index("CHECK_CLOSED")


def test_the_next_viewing_distance_is_tried_when_navigation_fails():
    io = FakeIO(robot=robot_at(1.0, 4.0), navigations=[False, True] + [True] * 5)

    outcome, _ = run(io)

    assert outcome == SUCCEEDED
    views = [call[1] for call in io.calls if call[0] == "navigate"][:2]
    assert [round(float(v[1, 3]), 2) for v in views] == [2.0, 1.8]


def test_no_reachable_viewing_pose_stops():
    io = FakeIO(robot=robot_at(1.0, 4.0), navigations=[False, False])

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "view the door" in userdata.message
    assert "plan_approach" not in io.names()


def test_a_view_lock_that_never_agrees_stops():
    io = FakeIO()
    answers = iter([[sample()] * 5, scattered(), scattered()])
    io.door_samples = lambda: next(answers)

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "spread" in userdata.message
    assert "plan_approach" not in io.names()


def test_view_range_is_close_and_near_the_normal():
    door = sample()

    assert view_poses(door, [1.2])[0][:2, 3] == pytest.approx([0.0, 2.0])
    assert in_view_range(door, robot_at(0.3, 1.8), 1.5, np.radians(35.0))
    assert not in_view_range(door, robot_at(0.0, 3.0), 1.5, np.radians(35.0))
    assert not in_view_range(door, robot_at(1.0, 1.0), 1.5, np.radians(35.0))


LOST = "localization lost: the scan matching on /odom_icp has failed for 0.6 s; the pose follows the wheels"


def test_a_lost_localization_while_opening_stops_without_trying_again():
    io = FakeIO(openings=[False, True], open_message=LOST)

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert "localization lost" in userdata.message
    assert io.names().count("open_door") == 1
    assert "stop_all" in io.names()


def test_a_lost_localization_while_navigating_tries_no_other_approach_pose():
    io = FakeIO(navigations=[False, True], nav_message=LOST)

    outcome, userdata = run(io)

    assert outcome == FAILED
    assert io.names().count("navigate") == 1
    assert "stop_all" in io.names()
