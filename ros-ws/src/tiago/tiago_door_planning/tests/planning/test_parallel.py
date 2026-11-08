import os

from tiago_door_planning.planning.parallel import ForkedMap


def test_results_keep_the_order_of_the_arguments():
    square = ForkedMap(lambda x: x * x, 3)
    try:
        assert square(list(range(20))) == [x * x for x in range(20)]
    finally:
        square.close()


def test_work_runs_in_other_processes():
    process_id = ForkedMap(lambda _: os.getpid(), 2)
    try:
        assert os.getpid() not in process_id(list(range(8)))
    finally:
        process_id.close()


def test_one_worker_runs_in_this_process():
    process_id = ForkedMap(lambda _: os.getpid(), 1)

    assert process_id([0, 1]) == [os.getpid(), os.getpid()]
