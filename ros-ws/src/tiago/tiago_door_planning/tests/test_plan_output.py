import numpy as np
import pytest

from tiago_door_planning.door_planner import PathStep
from tiago_door_planning.plan_output import plan_to_segments, split_by_grasp

START = PathStep([np.array([0.0, 0.0, 0.0])], [0.0], 0.0, True, np.zeros(2))


def move(x_end, angle_end, grasped, duration=1.0):
    poses = [np.array([x_end - 0.05, 0.0, 0.0]), np.array([x_end, 0.0, 0.0])]
    return PathStep(poses, [angle_end, angle_end], duration, grasped, np.ones(2) * x_end)


def regrasp(x, angle, grasped):
    return PathStep([np.array([x, 0.0, 0.0])], [angle], 7.5, grasped, np.full(2, 9.0))


def test_split_groups_consecutive_modes():
    steps = [START, move(0.1, 0.1, True), move(0.2, 0.2, True), regrasp(0.2, 0.2, False), move(0.3, 0.2, False)]

    assert [(grasped, len(run)) for grasped, run in split_by_grasp(steps)] == [(True, 2), (False, 2)]


def test_segments_skip_regrasp_and_resample_uniformly():
    steps = [START, move(0.1, 0.1, True), regrasp(0.1, 0.1, False), move(0.2, 0.1, False),
             regrasp(0.2, 0.1, True), move(0.3, 0.2, True)]

    segments = plan_to_segments(steps, "left", 0.25)

    assert [segment.grasped for segment in segments] == [True, False, True]
    first = segments[0].trajectory
    assert first.base_poses[0] == pytest.approx([0.0, 0.0, 0.0])
    assert first.base_poses[-1] == pytest.approx([0.1, 0.0, 0.0])
    assert len(first.door_angles) == 5


def test_grasped_segment_after_regrasp_starts_from_grasp_configuration():
    steps = [START, move(0.1, 0.1, True), regrasp(0.1, 0.1, False), move(0.2, 0.1, False),
             regrasp(0.2, 0.1, True), move(0.3, 0.2, True)]

    last = plan_to_segments(steps, "left", 0.25)[-1].trajectory

    assert last.elbow_hint == pytest.approx(np.full(2, 9.0))
    assert last.base_poses[0] == pytest.approx([0.2, 0.0, 0.0])
