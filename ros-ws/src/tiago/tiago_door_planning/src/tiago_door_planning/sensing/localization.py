"""Whether the SLAM still tracks."""

import collections

LOCALIZATION_LOST = "localization lost"

LocalizationParams = collections.namedtuple(
    "LocalizationParams", "topic lost_covariance max_lost_s max_silence_s"
)


class LocalizationWatch(object):
    """Fed with (arrival time, pose variance) of each odometry message; lost when the matcher has failed for longer than
    max_lost_s or gone silent for longer than max_silence_s. Times are arrival times, so a robot clock running off
    the laptop's does not matter."""

    def __init__(self, params, now):
        self._params = params
        self._started = now
        self._last = None
        self._lost_since = None

    def update(self, now, variance):
        self._last = now
        if variance < self._params.lost_covariance:
            self._lost_since = None
        elif self._lost_since is None:
            self._lost_since = now

    def lost(self, now):
        """The reason the localization cannot be trusted, starting with LOCALIZATION_LOST; None while it tracks."""
        params = self._params
        if self._last is None:
            if now - self._started > params.max_silence_s:
                return "%s: nothing on %s" % (LOCALIZATION_LOST, params.topic)
            return None
        if now - self._last > params.max_silence_s:
            return "%s: nothing on %s for %.1f s" % (
                LOCALIZATION_LOST,
                params.topic,
                now - self._last,
            )
        if self._lost_since is not None and now - self._lost_since > params.max_lost_s:
            return (
                "%s: the scan matching on %s has failed for %.1f s; the pose follows the wheels"
                % (
                    LOCALIZATION_LOST,
                    params.topic,
                    now - self._lost_since,
                )
            )
        return None


def is_localization_lost(message):
    return message.startswith(LOCALIZATION_LOST)
