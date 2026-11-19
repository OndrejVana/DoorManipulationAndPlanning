import numpy as np
import pytest

from tiago_door_planning.model.geometry import rotation_z
from tiago_door_planning.sensing.wrench import (
    ForceParams,
    contact_start,
    contact_update,
    filter_wrench,
    force_in_frame,
    grasp_lost_by_force,
    lost_grasp_start,
    lost_grasp_update,
    in_contact,
    low_pass,
    zero_offset,
)

PARAMS = ForceParams("/wrist_{side}_ft/corrected", 5.0, 0.3, 0.04, 6.0, 3, 2.0, 0.3, ["push"])


def test_zero_offset_is_the_mean_of_the_samples():
    samples = [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3], [3.0, 2.0, 1.0, 0.3, 0.2, 0.1]]

    assert np.allclose(zero_offset(samples), [2.0, 2.0, 2.0, 0.2, 0.2, 0.2])


def test_zero_offset_needs_samples():
    with pytest.raises(ValueError):
        zero_offset([])


def test_low_pass_starts_at_the_first_sample_and_moves_toward_the_next():
    first = low_pass(None, [1.0, 0.0], 0.02, 0.04)
    second = low_pass(first, [4.0, 0.0], 0.02, 0.04)

    assert np.allclose(first, [1.0, 0.0])
    assert np.allclose(second, [2.0, 0.0])


def test_low_pass_with_no_time_constant_follows_the_sample():
    assert np.allclose(low_pass(np.zeros(2), [5.0, 1.0], 0.02, 0.0), [5.0, 1.0])


def test_force_in_frame_rotates_force_and_torque():
    wrench = [1.0, 0.0, 2.0, 0.0, 1.0, 0.0]

    rotated = force_in_frame(rotation_z(np.pi / 2.0)[:3, :3], wrench)

    assert np.allclose(rotated, [0.0, 1.0, 2.0, -1.0, 0.0, 0.0])


def run(forces, zero=np.zeros(6), period=0.02):
    state = contact_start()
    for index, force in enumerate(forces):
        state = contact_update(state, index * period, [force, 0, 0, 0, 0, 0], zero, PARAMS)
    return state


def test_a_force_above_the_threshold_for_enough_samples_is_contact():
    state = run([1.0] * 5 + [20.0] * 6)

    assert in_contact(state, PARAMS)
    assert state.peak > 6.0


def test_a_single_spike_is_not_contact():
    state = run([1.0] * 5 + [40.0] + [1.0] * 10)

    assert not in_contact(state, PARAMS)
    assert state.above == 0


def test_contact_is_measured_from_the_zero():
    zero = np.array([30.0, 0, 0, 0, 0, 0])

    assert not in_contact(run([31.0] * 10, zero), PARAMS)
    assert in_contact(run([45.0] * 10, zero), PARAMS)


def test_filter_wrench_zeroes_and_filters_the_samples_in_order():
    zero = np.array([1.0, 0, 0, 0, 0, 0])
    samples = [(0.0, [2.0, 0, 0, 0, 0, 0]), (0.02, [5.0, 0, 0, 0, 0, 0])]

    filtered, stamp = filter_wrench(None, None, samples, zero, 0.04)

    assert stamp == 0.02
    assert np.allclose(filtered, [2.0, 0, 0, 0, 0, 0])


def test_filter_wrench_without_samples_keeps_its_state():
    assert filter_wrench(None, 1.0, [], np.zeros(6), 0.04) == (None, 1.0)


def run_loads(loads, advancing=True, period=0.05):
    state = lost_grasp_start()
    for index, load in enumerate(loads):
        state = lost_grasp_update(state, index * period, load, advancing, 2.0)
    return state, (len(loads) - 1) * period


def test_no_loss_while_the_load_builds_up():
    state, now = run_loads([0.0, 0.5, 1.0, 1.5, 3.0, 3.5, 3.9])

    assert not state.armed
    assert not grasp_lost_by_force(state, now, 0.3)


def test_a_load_that_collapses_for_long_enough_is_a_loss():
    state, now = run_loads([5.0, 8.0, 8.0] + [0.5] * 8)

    assert grasp_lost_by_force(state, now, 0.3)
    assert state.lowest == 0.5


def test_a_short_dip_is_no_loss():
    state, now = run_loads([8.0, 1.0, 1.0, 8.0, 1.0, 1.0])

    assert not grasp_lost_by_force(state, now, 0.3)


def test_no_loss_while_the_reference_stands_still():
    state, now = run_loads([8.0] + [0.0] * 10, advancing=False)

    assert not grasp_lost_by_force(state, now, 0.3)
