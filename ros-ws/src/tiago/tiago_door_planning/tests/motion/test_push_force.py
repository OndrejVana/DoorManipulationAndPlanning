from tiago_door_planning.motion.compliance import TANGENTIAL
from tiago_door_planning.motion.push_force import (
    PushForceParams,
    breakaway_start,
    breakaway_update,
    push_rate,
    resisting_force,
)

PARAMS = PushForceParams(slow=25.0, stop=50.0, lost_force=2.0, lost_s=0.3, hold_s=0.5)


def test_plan_runs_at_full_rate_below_slow():
    assert push_rate(breakaway_start(), 0.0, 10.0, PARAMS) == 1.0


def test_plan_slows_between_slow_and_stop():
    assert push_rate(breakaway_start(), 0.0, 37.5, PARAMS) == 0.5


def test_plan_stops_at_stop():
    assert push_rate(breakaway_start(), 0.0, 60.0, PARAMS) == 0.0


def test_sudden_drop_after_a_hard_push_holds_the_plan():
    state = breakaway_update(breakaway_start(), 0.0, 40.0, PARAMS)
    state = breakaway_update(state, 0.2, 1.0, PARAMS)

    assert push_rate(state, 0.3, 1.0, PARAMS) == 0.0
    assert push_rate(state, 0.8, 1.0, PARAMS) == 1.0


def test_slow_release_is_no_breakaway():
    state = breakaway_update(breakaway_start(), 0.0, 40.0, PARAMS)
    state = breakaway_update(state, 1.0, 1.0, PARAMS)

    assert push_rate(state, 1.0, 1.0, PARAMS) == 1.0


def test_light_push_dropping_is_no_breakaway():
    state = breakaway_update(breakaway_start(), 0.0, 10.0, PARAMS)
    state = breakaway_update(state, 0.1, 0.0, PARAMS)

    assert push_rate(state, 0.2, 0.0, PARAMS) == 1.0


def test_resisting_force_is_the_door_pushing_back_against_its_motion():
    assert resisting_force([3.0, -20.0, 1.0], TANGENTIAL) == 20.0
    assert resisting_force([3.0, 20.0, 1.0], TANGENTIAL) == 0.0
