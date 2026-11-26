"""How fast a push segment may advance given how hard the door pushes back on the hand."""

import collections

from .tracking import slowdown_factor

# slow / stop: resisting force (N) where the plan starts to slow / stops; lost_force, lost_s: a resisting force that
# fell from above slow to below lost_force within lost_s is a breakaway; hold_s: how long the plan then holds
PushForceParams = collections.namedtuple(
    "PushForceParams", "slow stop lost_force lost_s hold_s"
)

# last_high: last time the resisting force was at or above slow; hold_until: the plan holds until then
Breakaway = collections.namedtuple("Breakaway", "last_high hold_until")


def breakaway_start():
    return Breakaway(None, None)


def breakaway_update(state, now, resisting, params):
    """state after a resisting force (N, >= 0 while the door pushes back on the hand) measured at now."""
    if resisting >= params.slow:
        return Breakaway(now, state.hold_until)
    if (
        state.last_high is not None
        and resisting < params.lost_force
        and now - state.last_high <= params.lost_s
    ):
        return Breakaway(None, now + params.hold_s)
    return state


def push_rate(state, now, resisting, params):
    """Factor (0..1) on the plan's progress rate: 0 while holding after a breakaway, else easing off from slow to 0 at
    stop."""
    if state.hold_until is not None and now < state.hold_until:
        return 0.0
    return slowdown_factor(resisting, params.slow, params.stop)


def resisting_force(door_force, tangential):
    """The door pushing back on the hand against its motion; door_force is the force on the hand in door axes."""
    return max(0.0, -float(door_force[tangential]))
