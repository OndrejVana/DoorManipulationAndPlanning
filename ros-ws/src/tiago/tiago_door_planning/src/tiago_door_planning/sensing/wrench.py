import collections

import numpy as np

ForceParams = collections.namedtuple(
    "ForceParams",
    "topic wait_timeout_s zero_s filter_s contact_force contact_samples lost_force lost_s lost_by_force",
)

ContactState = collections.namedtuple("ContactState", "filtered stamp above peak")

LostGrasp = collections.namedtuple("LostGrasp", "armed since lowest")


def zero_offset(samples):
    """The residual a calibrated sensor still reads, averaged while the hand touches nothing."""
    samples = np.asarray(samples, dtype=float)
    if len(samples) == 0:
        raise ValueError("no wrench samples to zero on")
    return samples.mean(axis=0)


def low_pass(filtered, sample, dt, time_constant):
    """First-order filter; the first sample passes through unchanged."""
    sample = np.asarray(sample, dtype=float)
    if filtered is None:
        return sample
    return filtered + dt / (time_constant + dt) * (sample - filtered)


def filter_wrench(filtered, stamp, samples, zero, time_constant):
    """(filtered, stamp) after the zeroed (stamp, wrench) samples in order."""
    for sample_stamp, wrench in samples:
        dt = 0.0 if stamp is None else max(sample_stamp - stamp, 0.0)
        filtered = low_pass(filtered, np.asarray(wrench) - zero, dt, time_constant)
        stamp = sample_stamp
    return filtered, stamp


def force_in_frame(rotation, wrench):
    """The wrench rotated into another frame, rotation mapping sensor-frame vectors into it."""
    wrench = np.asarray(wrench, dtype=float)
    return np.concatenate([rotation @ wrench[:3], rotation @ wrench[3:]])


def contact_start():
    return ContactState(None, None, 0, 0.0)


def contact_update(state, stamp, wrench, zero, params):
    """Filters the zeroed force and counts consecutive samples above the contact force: one spike is noise."""
    dt = 0.0 if state.stamp is None else max(stamp - state.stamp, 0.0)
    filtered = low_pass(state.filtered, np.asarray(wrench) - zero, dt, params.filter_s)
    magnitude = float(np.linalg.norm(filtered[:3]))
    above = state.above + 1 if magnitude > params.contact_force else 0
    return ContactState(filtered, stamp, above, max(state.peak, magnitude))


def in_contact(state, params):
    return state.above >= params.contact_samples


def lost_grasp_start():
    return LostGrasp(False, None, float("inf"))


def lost_grasp_update(state, now, load, advancing, lost_force):
    """Tracks the force between the door and the hand along the door's motion (load >= 0). It counts once it has reached twice lost_force, so the
    build-up at the start of a segment is no loss; then a load below lost_force while the reference still advances
    starts the clock, and anything else stops it."""
    armed = state.armed or load >= 2.0 * lost_force
    lowest = min(state.lowest, load) if armed else state.lowest
    if not (armed and advancing and load < lost_force):
        return LostGrasp(armed, None, lowest)
    return LostGrasp(armed, now if state.since is None else state.since, lowest)


def grasp_lost_by_force(state, now, lost_s):
    """The door has not pushed back on the hand for lost_s: it ran ahead of the palm or the hand slipped off."""
    return state.since is not None and now - state.since >= lost_s
