import multiprocessing
import signal
import threading

_task = None


def _ignore_signals():
    """The forked workers carry the ROS node's signal handlers; the pool alone ends them."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _run(argument):
    return _task(argument)


class ForkedMap(object):
    """task over a list of arguments, on forked workers once the first list is long enough to share."""

    def __init__(self, task, workers):
        self.task = task
        self.workers = workers
        self.pool = None

    def __call__(self, arguments):
        if self.workers <= 1 or len(arguments) < 2:
            return [self.task(argument) for argument in arguments]
        if self.pool is None:
            self.pool = self._fork()
        chunk = -(-len(arguments) // self.workers)
        return self.pool.map(_run, arguments, chunksize=chunk)

    def _fork(self):
        global _task
        _task = self.task
        return multiprocessing.get_context("fork").Pool(
            self.workers, initializer=_ignore_signals
        )

    def close(self):
        """Waiting for the workers to exit takes a tenth of a second, so a background thread does it."""
        if self.pool is not None:
            self.pool.close()
            threading.Thread(target=self.pool.join, daemon=True).start()
            self.pool = None
