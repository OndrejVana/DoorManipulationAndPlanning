import collections

SlowCycle = collections.namedtuple("SlowCycle", "total parts")


class CycleTimer(object):
    """start() opens a cycle, mark(name, now) books the time since the previous mark to name, and the next start()
    closes the cycle; a cycle longer than slow_s is returned by start() as a SlowCycle. Times are wall-clock seconds.
    """

    def __init__(self, slow_s):
        self._slow_s = slow_s
        self._start = None
        self._last = None
        self._parts = collections.OrderedDict()
        self._cycles = 0
        self._slow = 0
        self._max_total = 0.0
        self._max_parts = collections.OrderedDict()

    def start(self, now):
        slow = self._close(now)
        self._start = self._last = now
        self._parts = collections.OrderedDict()
        return slow

    def mark(self, name, now):
        self._parts[name] = self._parts.get(name, 0.0) + now - self._last
        self._last = now

    def _close(self, now):
        if self._start is None:
            return None
        if now > self._last:
            self.mark("other", now)
        total = now - self._start
        self._cycles += 1
        self._max_total = max(self._max_total, total)
        for name, spent in self._parts.items():
            self._max_parts[name] = max(self._max_parts.get(name, 0.0), spent)
        if total <= self._slow_s:
            return None
        self._slow += 1
        return SlowCycle(total, dict(self._parts))

    def summary(self):
        """(cycles, slow cycles, longest cycle, longest time per part) over the closed cycles."""
        return self._cycles, self._slow, self._max_total, dict(self._max_parts)


def describe_parts(parts):
    """The parts in milliseconds, longest first."""
    ordered = sorted(parts.items(), key=lambda item: -item[1])
    return ", ".join("%s %.0f" % (name, 1000.0 * spent) for name, spent in ordered)
