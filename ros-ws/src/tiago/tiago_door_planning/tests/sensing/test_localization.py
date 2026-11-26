from tiago_door_planning.sensing.localization import (
    LocalizationParams,
    LocalizationWatch,
    is_localization_lost,
)

PARAMS = LocalizationParams(topic="/odom_icp", lost_covariance=1000.0, max_lost_s=0.5, max_silence_s=1.0)
TRACKED, LOST = 2e-5, 9999.0


def feed(watch, start, end, variance, period=0.07):
    t = start
    while t <= end + 1e-9:
        watch.update(t, variance)
        t += period


def test_tracking_matcher_is_not_lost():
    watch = LocalizationWatch(PARAMS, 0.0)
    feed(watch, 0.0, 5.0, TRACKED)

    assert watch.lost(5.0) is None


def test_short_matching_failure_is_bridged():
    watch = LocalizationWatch(PARAMS, 0.0)
    feed(watch, 0.0, 2.0, TRACKED)
    feed(watch, 2.07, 2.4, LOST)

    assert watch.lost(2.4) is None


def test_matching_failure_longer_than_max_lost_s_is_lost():
    watch = LocalizationWatch(PARAMS, 0.0)
    feed(watch, 0.0, 2.0, TRACKED)
    feed(watch, 2.07, 2.7, LOST)

    reason = watch.lost(2.7)

    assert reason is not None and is_localization_lost(reason)


def test_one_matched_frame_ends_the_failure():
    watch = LocalizationWatch(PARAMS, 0.0)
    feed(watch, 0.0, 0.4, LOST)
    watch.update(0.45, TRACKED)
    feed(watch, 0.5, 0.9, LOST)

    assert watch.lost(0.9) is None


def test_silent_topic_is_lost():
    watch = LocalizationWatch(PARAMS, 0.0)
    feed(watch, 0.0, 1.0, TRACKED)

    assert is_localization_lost(watch.lost(2.5))


def test_no_message_is_lost_only_after_max_silence_s():
    watch = LocalizationWatch(PARAMS, 10.0)

    assert watch.lost(10.5) is None
    assert is_localization_lost(watch.lost(11.5))
