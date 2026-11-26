from tiago_door_planning.motion.cycle_timer import CycleTimer, describe_parts


def run_cycle(timer, start, parts):
    """One cycle starting at start with (name, seconds) parts; returns the time it ended."""
    timer.start(start)
    now = start
    for name, spent in parts:
        now += spent
        timer.mark(name, now)
    return now


def test_a_fast_cycle_is_not_reported():
    timer = CycleTimer(0.04)
    end = run_cycle(timer, 0.0, [("measure", 0.005), ("sleep", 0.015)])

    assert timer.start(end) is None


def test_a_slow_cycle_reports_its_parts():
    timer = CycleTimer(0.04)
    end = run_cycle(timer, 0.0, [("measure", 0.005), ("laser", 0.2), ("sleep", 0.01)])

    slow = timer.start(end)

    assert abs(slow.total - 0.215) < 1e-9
    assert abs(slow.parts["laser"] - 0.2) < 1e-9


def test_time_after_the_last_mark_counts_as_other():
    timer = CycleTimer(0.04)
    end = run_cycle(timer, 0.0, [("measure", 0.005)])

    slow = timer.start(end + 0.1)

    assert abs(slow.parts["other"] - 0.1) < 1e-9


def test_summary_counts_cycles_and_keeps_the_longest_part():
    timer = CycleTimer(0.04)
    end = run_cycle(timer, 0.0, [("laser", 0.01)])
    end = run_cycle(timer, end, [("laser", 0.3)])
    timer.start(end)

    cycles, slow, longest, parts = timer.summary()

    assert (cycles, slow) == (2, 1)
    assert abs(longest - 0.3) < 1e-9
    assert abs(parts["laser"] - 0.3) < 1e-9


def test_parts_are_described_longest_first_in_ms():
    assert describe_parts({"measure": 0.002, "laser": 0.12}) == "laser 120, measure 2"
